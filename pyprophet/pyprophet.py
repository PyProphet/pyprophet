# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile  # ignore
except:
    profile = lambda x: x

import pandas as pd
import numpy as np

from stats import (lookup_s_and_q_values_from_error_table, calculate_final_statistics,
                   mean_and_std_dev, final_err_table, summary_err_table)
from config import CONFIG

from data_handling import (prepare_data_table, Experiment)
from classifiers import (LDALearner)
from semi_supervised import (AbstractSemiSupervisedLearner, StandardSemiSupervisedLearner)

import multiprocessing

import logging

import time


def unwrap_self_for_multiprocessing((inst, method_name, args),):
    """ You can not call methods with multiprocessing, but free functions,
        If you want to call  inst.method(arg0, arg1),

            unwrap_self_for_multiprocessing(inst, "method", (arg0, arg1))

        does the trick.
    """
    return getattr(inst, method_name)(*args)


class HolyGostQuery(object):

    """ HolyGhostQuery assembles the unsupervised methods.
        See below how PyProphet parameterises this class.
    """

    def __init__(self, semi_supervised_learner):
        assert isinstance(semi_supervised_learner,
                          AbstractSemiSupervisedLearner)
        self.semi_supervised_learner = semi_supervised_learner

    @profile
    def process_csv(self, path, delim=",", loaded_scorer=None):
        start_at = time.time()

        logging.info("read %s" % path)

        table = pd.read_csv(path, delim, na_values=["NA", "NaN", "infinite"])

        if loaded_scorer is not None:
            logging.info("apply scorer to  %s" % path)
            result_tables, data_for_persistence = self.apply_loaded_scorer(table, loaded_scorer)
            data_for_persistence = None
        else:
            logging.info("learn and apply scorer to %s" % path)
            result_tables, data_for_persistence = self.learn_and_apply_classifier(table)
        logging.info("processing %s finished" % path)

        needed = time.time() - start_at
        hours = int(needed / 3600)
        needed -= hours * 3600

        minutes = int(needed / 60)
        needed -= minutes * 60

        logging.info("time needed: %02d:%02d:%.1f" % (hours, minutes, needed))
        return result_tables, data_for_persistence

    @profile
    def learn_and_apply_classifier(self, table):

        prepared_table, score_columns = prepare_data_table(table)

        experiment = Experiment(prepared_table)

        is_test = CONFIG.get("is_test", False)

        if is_test:  # for reliable results
            experiment.df.sort("tg_id", ascending=True, inplace=True)

        experiment.log_summary()

        all_test_target_scores = []
        all_test_decoy_scores = []
        ws = []
        neval = CONFIG.get("xeval.num_iter")
        inst = self.semi_supervised_learner
        num_processes = CONFIG.get("num_processes")
        logging.info("start %d cross evals using %d processes" % (neval, num_processes))
        if num_processes == 1:
            for k in range(neval):
                (ttt_scores, ttd_scores, w) = inst.learn_randomized(experiment)
                all_test_target_scores.extend(ttt_scores)
                all_test_decoy_scores.extend(ttd_scores)
                ws.append(w.flatten())
        else:
            pool = multiprocessing.Pool(processes=num_processes)
            while neval:
                remaining = max(0, neval - num_processes)
                todo = neval - remaining
                neval -= todo
                args = ((inst, "learn_randomized", (experiment, )), ) * todo
                res = pool.map(unwrap_self_for_multiprocessing, args)
                top_test_target_scores = [ti for r in res for ti in r[0]]
                top_test_decoy_scores = [ti for r in res for ti in r[1]]
                ws.extend([r[2] for r in res])
                all_test_target_scores.extend(top_test_target_scores)
                all_test_decoy_scores.extend(top_test_decoy_scores)

        logging.info("finished cross evals")
        final_classifier = self.semi_supervised_learner.averaged_learner(ws)

        result, data_for_persistence = self.apply_classifier(final_classifier, experiment,
                                                             all_test_target_scores,
                                                             all_test_decoy_scores, table)
        logging.info("calculated scoring and statistics")
        return result, data_for_persistence + (score_columns,)

    @profile
    def apply_loaded_scorer(self, table, loaded_scorer):

        final_classifier, mu, nu, df_raw_stat, loaded_score_columns = loaded_scorer

        prepared_table, __ = prepare_data_table(table, loaded_score_columns=loaded_score_columns)

        experiment = Experiment(prepared_table)

        final_score = final_classifier.score(experiment, True)
        experiment["d_score"] = (final_score - mu) / nu

        scored_table = self.enrich_table_with_results(table, experiment, df_raw_stat)

        return (None, None, scored_table), None

    @profile
    def enrich_table_with_results(self, table, experiment, df_raw_stat):
        s_values, q_values = lookup_s_and_q_values_from_error_table(experiment["d_score"],
                                                                    df_raw_stat)
        experiment["m_score"] = q_values
        experiment["s_value"] = s_values
        logging.info("mean m_score = %e, std_dev m_score = %e" % (np.mean(q_values),
                     np.std(q_values, ddof=1)))
        logging.info("mean s_value = %e, std_dev s_value = %e" % (np.mean(s_values),
                     np.std(s_values, ddof=1)))
        experiment.add_peak_group_rank()

        scored_table = table.join(experiment[["d_score", "m_score", "peak_group_rank"]])
        return scored_table

    @profile
    def apply_classifier(self, final_classifier, experiment, all_test_target_scores,
                         all_test_decoy_scores, table):

        lambda_ = CONFIG.get("final_statistics.lambda")

        mu, nu, final_score = self.calculate_params_for_d_score(final_classifier, experiment)
        experiment["d_score"] = (final_score - mu) / nu

        all_tt_scores = experiment.get_top_target_peaks()["d_score"]

        df_raw_stat = calculate_final_statistics(all_tt_scores, all_test_target_scores,
                                                 all_test_decoy_scores, lambda_)

        scored_table = self.enrich_table_with_results(table, experiment, df_raw_stat)

        final_statistics = final_err_table(df_raw_stat)
        summary_statistics = summary_err_table(df_raw_stat)

        needed_to_persist = (final_classifier, mu, nu,
                             df_raw_stat.loc[:, ["svalue", "qvalue", "cutoff"]])
        return (summary_statistics, final_statistics, scored_table), needed_to_persist

    @profile
    def calculate_params_for_d_score(self, classifier, experiment):
        score = classifier.score(experiment, True)
        experiment.set_and_rerank("classifier_score", score)
        td_scores = experiment.get_top_decoy_peaks()["classifier_score"]
        mu, nu = mean_and_std_dev(td_scores)
        return mu, nu, score


@profile
def PyProphet():
    return HolyGostQuery(StandardSemiSupervisedLearner(LDALearner()))
