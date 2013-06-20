#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except:
    profile = lambda x: x

import pandas as pd
import random

from stats import (lookup_q_values_from_error_table,
                   calculate_final_statistics, mean_and_std_dev,
                   final_err_table, summary_err_table)

from data_handling import (prepare_data_table, Experiment)
from classifiers import (LDALearner)
from semi_supervised import (AbstractSemiSupervisedLearner, StandardSemiSupervisedLearner)

import multiprocessing


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
    def process_csv(self, path, delim=",", config=None):
        if config is None:
            config = dict()
        table = pd.read_csv(path, delim)
        return self.learn_and_apply_classifier(table, config)

    @profile
    def learn_and_apply_classifier(self, table, config):

        experiment = Experiment(prepare_data_table(table))

        is_test = config.get("is_test", False)

        if is_test: # for reliable results
            experiment.df.sort("tg_id", ascending=True, inplace=True)
            random.seed(0)

        experiment.print_summary()

        all_test_target_scores = []
        all_test_decoy_scores = []
        ws = []
        neval = 1
        inst = self.semi_supervised_learner
        num_processes = config.get("xeval.num_processes")
        if num_processes == 1:
            for k in range(neval):
                (ttt_scores, ttd_scores, w) = inst.learn(experiment, config)
                all_test_target_scores.extend(ttt_scores)
                all_test_decoy_scores.extend(ttd_scores)
                ws.append(w.flatten())
        else:

            pool = multiprocessing.Pool(processes=num_processes)
            while neval:
                remaining = max(0, neval-num_processes)
                todo = neval-remaining
                neval -= todo
                args= ( (inst, "learn", (experiment, config)), ) * todo
                res = pool.map(unwrap_self_for_multiprocessing, args)
                top_test_target_scores = [ ti for r in res for ti in r[0]]
                top_test_decoy_scores = [ ti for r in res for ti in r[1]]
                ws.extend([ r[2] for r in res])
                all_test_target_scores.extend(top_test_target_scores)
                all_test_decoy_scores.extend(top_test_decoy_scores)

        final_classifier = self.semi_supervised_learner.averaged_learner(ws)
        return self.apply_classifier(final_classifier,
                                     experiment,
                                     config,
                                     all_test_target_scores,
                                     all_test_decoy_scores,
                                     table)

    @profile
    def apply_classifier(self,
                         final_classifier,
                         experiment,
                         config,
                         all_test_target_scores,
                         all_test_decoy_scores,
                         table):

        lambda_ = config.get("final_statistics.lambda")

        self.assign_d_score(final_classifier, experiment)

        all_tt_scores  = experiment.get_top_target_peaks()["d_score"]


        df_raw_stat = calculate_final_statistics(all_tt_scores,
                                                 all_test_target_scores,
                                                 all_test_decoy_scores,
                                                 lambda_)

        final_statistics = final_err_table(df_raw_stat)
        summary_statistics = summary_err_table(df_raw_stat)

        q_values = lookup_q_values_from_error_table(experiment["classifier_score"], df_raw_stat)
        experiment["q_values"] = q_values

        # as experiment maybe permutated row wise, directly attaching q_values
        # to table might result in wrong assignment:
        scored_table = table.join(experiment[["d_score", "q_values"]])

        return summary_statistics, final_statistics, scored_table

    @profile
    def assign_d_score(self, final_classifier, experiment):

        # with .values it's faster:
        final_score = final_classifier.score(experiment, True)

        experiment.set_and_rerank("classifier_score", final_score)

        td_scores = experiment.get_top_decoy_peaks()["classifier_score"]
        mu, nu = mean_and_std_dev(td_scores)
        experiment["d_score"] =(final_score - mu)/nu


def PyProphet():
    return HolyGostQuery(StandardSemiSupervisedLearner(LDALearner()))

