# encoding: latin-1

from __future__ import division

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    def profile(fun):
        return fun

import pandas as pd
import numpy as np

from .config import CONFIG, set_pandas_print_options

from .stats import (lookup_values_from_error_table, error_statistics,
                   mean_and_std_dev, final_err_table, summary_err_table,
                   posterior_chromatogram_hypotheses_fast)

from .data_handling import (prepare_data_table, Experiment)
from .classifiers import (LDALearner)
from .semi_supervised import (AbstractSemiSupervisedLearner, StandardSemiSupervisedLearner)

import multiprocessing

from .std_logger import logging

import time

from collections import namedtuple
from contextlib import contextmanager
import sys

@contextmanager
def timer(name=""):
    start_at = time.time()

    yield

    needed = time.time() - start_at
    hours = int(needed / 3600)
    needed -= hours * 3600

    minutes = int(needed / 60)
    needed -= minutes * 60

    if name:
        logging.info("time needed for %s: %02d:%02d:%.1f" % (name, hours, minutes, needed))
    else:
        logging.info("time needed: %02d:%02d:%.1f" % (hours, minutes, needed))


Result = namedtuple("Result", "summary_statistics final_statistics scored_tables")


def unwrap_self_for_multiprocessing(arg):
    """ You can not call methods with multiprocessing, but free functions,
        If you want to call  inst.method(arg0, arg1),

            unwrap_self_for_multiprocessing(inst, "method", (arg0, arg1))

        does the trick.
    """
    (inst, method_name, args) = arg
    return getattr(inst, method_name)(*args)


@profile
def calculate_params_for_d_score(classifier, experiment):
    score = classifier.score(experiment, True)
    experiment.set_and_rerank("classifier_score", score)

    td_scores = experiment.get_top_decoy_peaks()["classifier_score"]

    mu, nu = mean_and_std_dev(td_scores)
    return mu, nu


class Scorer(object):

    def __init__(self, classifier, score_columns, experiment):

        self.classifier = classifier
        self.score_columns = score_columns
        self.mu, self.nu = calculate_params_for_d_score(classifier, experiment)
        final_score = classifier.score(experiment, True)
        experiment["d_score"] = (final_score - self.mu) / self.nu
        lambda_ = CONFIG.get("final_statistics.lambda")

        all_tt_scores = experiment.get_top_target_peaks()["d_score"]
        all_td_scores = experiment.get_top_decoy_peaks()["d_score"]

        use_pemp = CONFIG.get("final_statistics.emp_p")
        use_pfdr = CONFIG.get("final_statistics.pfdr")
        lfdr_trunc  = CONFIG.get("final_statistics.lfdr_trunc")
        lfdr_monotone  = CONFIG.get("final_statistics.lfdr_monotone")
        lfdr_transf  = CONFIG.get("final_statistics.lfdr_transf")
        lfdr_adj  = CONFIG.get("final_statistics.lfdr_adj")
        lfdr_eps  = CONFIG.get("final_statistics.lfdr_eps")

        pi0_method  = CONFIG.get("final_statistics.pi0_method")
        pi0_smooth_df  = CONFIG.get("final_statistics.pi0_smooth_df")
        pi0_smooth_log_pi0  = CONFIG.get("final_statistics.pi0_smooth_log_pi0")

        self.error_stat, self.pi0 = error_statistics(all_tt_scores,
                                                      all_td_scores,
                                                      lambda_,
                                                      pi0_method, 
                                                      pi0_smooth_df, 
                                                      pi0_smooth_log_pi0, 
                                                      use_pemp,
                                                      use_pfdr,
                                                      True,
                                                      lfdr_trunc,
                                                      lfdr_monotone,
                                                      lfdr_transf,
                                                      lfdr_adj,
                                                      lfdr_eps)

        self.number_target_pg = len(experiment.df[experiment.df.is_decoy.eq(False)])
        self.number_target_peaks = len(experiment.get_top_target_peaks().df)
        self.dvals = experiment.df.loc[(experiment.df.is_decoy.eq(True)), "d_score"]
        self.target_scores = experiment.get_top_target_peaks().df["d_score"]
        self.decoy_scores = experiment.get_top_decoy_peaks().df["d_score"]

    def score(self, table):

        prepared_table, __ = prepare_data_table(table, tg_id_name=CONFIG.get("group_id"), score_columns=self.score_columns)
        texp = Experiment(prepared_table)
        score = self.classifier.score(texp, True)
        texp["d_score"] = (score - self.mu) / self.nu

        p_values, s_values, peps, q_values = lookup_values_from_error_table(texp["d_score"].values,
                                                                    self.error_stat)

        texp["pep"] = peps
        texp["q_value"] = q_values
        texp["s_value"] = s_values
        texp["p_value"] = p_values
        logging.info("mean q_value = %e, std_dev q_value = %e" % (np.mean(q_values),
                                                                  np.std(q_values, ddof=1)))
        logging.info("mean s_value = %e, std_dev s_value = %e" % (np.mean(s_values),
                                                                  np.std(s_values, ddof=1)))
        texp.add_peak_group_rank()

        df = table.join(texp[["d_score", "p_value", "q_value", "pep", "peak_group_rank"]])

        if CONFIG.get("tric_chromprob"):
            df = self.add_chromatogram_probabilities(df, texp)

        return df

    def add_chromatogram_probabilities(self, scored_table, texp):
        prior_chrom_null = self.error_stat.num_null / self.error_stat.num_total
        allhypothesis, h0 = posterior_chromatogram_hypotheses_fast(texp, prior_chrom_null)
        texp.df["h_score"] = allhypothesis
        texp.df["h0_score"] = h0
        scored_table = scored_table.join(texp[["h_score", "h0_score"]])

        return scored_table

    def get_error_stats(self):
        return final_err_table(self.error_stat), summary_err_table(self.error_stat)

    def minimal_error_stat(self):
        minimal_err_stat = ErrorStatistics(self.error_stat.df.loc[:, ["svalue", "qvalue", "pvalue", "pep", "cutoff"]],
                                           self.error_stat.num_null, self.error_stat.num_total)
        return minimal_err_stat

    def __getstate__(self):
        """when pickling"""
        data = vars(self)
        data["error_stat"] = self.minimal_error_stat()
        return data

    def __setstate__(self, data):
        """when unpickling"""
        self.__dict__.update(data)

class HolyGostQuery(object):

    """ HolyGhostQuery assembles the unsupervised methods.
        See below how PyProphet parameterises this class.
    """

    def __init__(self, semi_supervised_learner):
        assert isinstance(semi_supervised_learner,
                          AbstractSemiSupervisedLearner)
        self.semi_supervised_learner = semi_supervised_learner

    def _setup_experiment(self, table):
        prepared_table, score_columns = prepare_data_table(table, tg_id_name=CONFIG.get("group_id"))
        experiment = Experiment(prepared_table)
        experiment.log_summary()
        return experiment, score_columns

    def apply_weights(self, table, loaded_weights):
        with timer():
            logging.info("apply weights")
            result, scorer, trained_weights = self._apply_weights(table, loaded_weights)
            logging.info("processing input data finished")
        return result, scorer, trained_weights

    def _apply_weights(self, table, loaded_weights):

        experiment, score_columns = self._setup_experiment(table)

        if np.all(score_columns == loaded_weights['score'].values):
            weights = loaded_weights['weight'].values
        else:
            sys.exit("Error: Scores in weights file do not match data.")

        final_classifier = self._apply_weights_on_exp(experiment, weights)

        return self._build_result(table, final_classifier, score_columns, experiment)

    def _apply_weights_on_exp(self, experiment, loaded_weights):

        learner = self.semi_supervised_learner

        logging.info("start application of pretrained weights")
        clf_scores = learner.score(experiment, loaded_weights)
        experiment.set_and_rerank("classifier_score", clf_scores)

        logging.info("finished pretrained scoring")

        ws = [loaded_weights.flatten()]
        final_classifier = self.semi_supervised_learner.averaged_learner(ws)

        return final_classifier

    @profile
    def learn_and_apply(self, table):
        with timer():

            logging.info("learn and apply classifier from input data")
            result, scorer, trained_weights = self._learn_and_apply(table)
            logging.info("processing input data finished")

        return result, scorer, trained_weights

    def _learn_and_apply(self, table):

        experiment, score_columns = self._setup_experiment(table)
        final_classifier = self._learn(experiment)

        return self._build_result(table, final_classifier, score_columns, experiment)

    def _learn(self, experiment):
        is_test = CONFIG.get("is_test")
        if is_test:  # for reliable results
            experiment.df.sort_values("tg_id", ascending=True, inplace=True)

        learner = self.semi_supervised_learner
        ws = []

        neval = CONFIG.get("xeval.num_iter")
        num_processes = CONFIG.get("num_processes")

        logging.info("learn and apply scorer")
        logging.info("start %d cross evals using %d processes" % (neval, num_processes))

        if num_processes == 1:
            for k in range(neval):
                (ttt_scores, ttd_scores, w) = learner.learn_randomized(experiment)
                ws.append(w.flatten())
        else:
            pool = multiprocessing.Pool(processes=num_processes)
            while neval:
                remaining = max(0, neval - num_processes)
                todo = neval - remaining
                neval -= todo
                args = ((learner, "learn_randomized", (experiment, )), ) * todo
                res = pool.map(unwrap_self_for_multiprocessing, args)
                ttt_scores = [ti for r in res for ti in r[0]]
                ttd_scores = [ti for r in res for ti in r[1]]
                ws.extend([r[2] for r in res])
        logging.info("finished cross evals")
        logging.info("")

        # we only use weights from last iteration
        ws = [ws[-1]]

        final_classifier = self.semi_supervised_learner.averaged_learner(ws)

        return final_classifier

    def _build_result(self, table, final_classifier, score_columns, experiment):

        weights = final_classifier.get_parameters()
        classifier_table = pd.DataFrame({'score': score_columns, 'weight': weights})

        scorer = Scorer(final_classifier, score_columns, experiment)

        scored_table = scorer.score(table)

        final_statistics, summary_statistics = scorer.get_error_stats()

        result = Result(summary_statistics, final_statistics, scored_table)

        logging.info("calculated scoring and statistics")
        return result, scorer, classifier_table

@profile
def PyProphet():
    return HolyGostQuery(StandardSemiSupervisedLearner(LDALearner()))
