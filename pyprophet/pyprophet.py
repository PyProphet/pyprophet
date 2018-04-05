from __future__ import division

import pandas as pd
import numpy as np
import multiprocessing
import sys
import time
import click

from .stats import (lookup_values_from_error_table, error_statistics,
                   mean_and_std_dev, final_err_table, summary_err_table,
                   posterior_chromatogram_hypotheses_fast)
from .data_handling import (prepare_data_table, Experiment)
from .classifiers import (LDALearner)
from .semi_supervised import (AbstractSemiSupervisedLearner, StandardSemiSupervisedLearner)
from collections import namedtuple
from contextlib import contextmanager

try:
    profile
except NameError:
    def profile(fun):
        return fun


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
        click.echo("Info: Time needed for %s: %02d:%02d:%.1f" % (name, hours, minutes, needed))
    else:
        click.echo("Info: Time needed: %02d:%02d:%.1f" % (hours, minutes, needed))


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

    def __init__(self, classifier, score_columns, experiment, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, tric_chromprob):

        self.classifier = classifier
        self.score_columns = score_columns
        self.mu, self.nu = calculate_params_for_d_score(classifier, experiment)
        final_score = classifier.score(experiment, True)
        experiment["d_score"] = (final_score - self.mu) / self.nu

        self.group_id = group_id
        self.parametric = parametric
        self.pfdr = pfdr
        self.pi0_lambda = pi0_lambda
        self.pi0_method = pi0_method
        self.pi0_smooth_df = pi0_smooth_df
        self.pi0_smooth_log_pi0 = pi0_smooth_log_pi0
        self.lfdr_truncate = lfdr_truncate
        self.lfdr_monotone = lfdr_monotone
        self.lfdr_transformation = lfdr_transformation
        self.lfdr_adj = lfdr_adj
        self.lfdr_eps = lfdr_eps
        self.tric_chromprob = tric_chromprob

        target_scores = experiment.get_top_target_peaks()["d_score"]
        decoy_scores = experiment.get_top_decoy_peaks()["d_score"]

        self.error_stat, self.pi0 = error_statistics(target_scores,
                                                     decoy_scores,
                                                     self.parametric,
                                                     self.pfdr,
                                                     self.pi0_lambda,
                                                     self.pi0_method, 
                                                     self.pi0_smooth_df, 
                                                     self.pi0_smooth_log_pi0, 
                                                     True, # compute_lfdr
                                                     self.lfdr_truncate,
                                                     self.lfdr_monotone,
                                                     self.lfdr_transformation,
                                                     self.lfdr_adj,
                                                     self.lfdr_eps)

        self.number_target_pg = len(experiment.df[experiment.df.is_decoy.eq(False)])
        self.number_target_peaks = len(experiment.get_top_target_peaks().df)
        self.dvals = experiment.df.loc[(experiment.df.is_decoy.eq(True)), "d_score"]
        self.target_scores = experiment.get_top_target_peaks().df["d_score"]
        self.decoy_scores = experiment.get_top_decoy_peaks().df["d_score"]

    def score(self, table):

        prepared_table, __ = prepare_data_table(table, tg_id_name=self.group_id, score_columns=self.score_columns)
        texp = Experiment(prepared_table)
        score = self.classifier.score(texp, True)
        texp["d_score"] = (score - self.mu) / self.nu

        p_values, s_values, peps, q_values = lookup_values_from_error_table(texp["d_score"].values,
                                                                    self.error_stat)

        texp["pep"] = peps
        texp["q_value"] = q_values
        texp["s_value"] = s_values
        texp["p_value"] = p_values
        click.echo("Info: Mean qvalue = %e, std_dev qvalue = %e" % (np.mean(q_values),
                                                                  np.std(q_values, ddof=1)))
        click.echo("Info: Mean svalue = %e, std_dev svalue = %e" % (np.mean(s_values),
                                                                  np.std(s_values, ddof=1)))
        texp.add_peak_group_rank()

        df = table.join(texp[["d_score", "p_value", "q_value", "pep", "peak_group_rank"]])

        if self.tric_chromprob:
            df = self.add_chromatogram_probabilities(df, texp)

        return df

    def add_chromatogram_probabilities(self, scored_table, texp):
        allhypothesis, h0 = posterior_chromatogram_hypotheses_fast(texp, self.pi0['pi0'])
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

    def __init__(self, semi_supervised_learner, ss_num_iter, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, tric_chromprob, threads, test):
        assert isinstance(semi_supervised_learner,
                          AbstractSemiSupervisedLearner)
        self.semi_supervised_learner = semi_supervised_learner
        self.ss_num_iter = ss_num_iter
        self.group_id = group_id
        self.parametric = parametric
        self.pfdr = pfdr
        self.pi0_lambda = pi0_lambda
        self.pi0_method = pi0_method
        self.pi0_smooth_df = pi0_smooth_df
        self.pi0_smooth_log_pi0 = pi0_smooth_log_pi0
        self.lfdr_truncate = lfdr_truncate
        self.lfdr_monotone = lfdr_monotone
        self.lfdr_transformation = lfdr_transformation
        self.lfdr_adj = lfdr_adj
        self.lfdr_eps = lfdr_eps
        self.tric_chromprob = tric_chromprob
        self.threads = threads
        self.test = test

    def _setup_experiment(self, table):
        prepared_table, score_columns = prepare_data_table(table, tg_id_name=self.group_id)
        experiment = Experiment(prepared_table)
        experiment.log_summary()
        return experiment, score_columns

    def apply_weights(self, table, loaded_weights):
        with timer():
            click.echo("Info: Applying weights.")
            result, scorer, trained_weights = self._apply_weights(table, loaded_weights)
            click.echo("Info: Finished processing of input data.")
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

        click.echo("Info: Start application of pretrained weights.")
        clf_scores = learner.score(experiment, loaded_weights)
        experiment.set_and_rerank("classifier_score", clf_scores)

        click.echo("Info: Finished pretrained scoring.")

        ws = [loaded_weights.flatten()]
        final_classifier = self.semi_supervised_learner.averaged_learner(ws)

        return final_classifier

    @profile
    def learn_and_apply(self, table):
        with timer():

            click.echo("Info: Learn and apply classifier from input data.")
            result, scorer, trained_weights = self._learn_and_apply(table)
            click.echo("Info: Processing input data finished.")

        return result, scorer, trained_weights

    def _learn_and_apply(self, table):

        experiment, score_columns = self._setup_experiment(table)
        final_classifier = self._learn(experiment)

        return self._build_result(table, final_classifier, score_columns, experiment)

    def _learn(self, experiment):
        if self.test:  # for reliable results
            experiment.df.sort_values("tg_id", ascending=True, inplace=True)

        learner = self.semi_supervised_learner
        ws = []

        neval = self.ss_num_iter

        click.echo("Info: Semi-supervised learning of weights:")
        click.echo("Info: Start learning on %d folds using %d processes." % (neval, self.threads))

        if self.threads == 1:
            for k in range(neval):
                (ttt_scores, ttd_scores, w) = learner.learn_randomized(experiment)
                ws.append(w.flatten())
        else:
            pool = multiprocessing.Pool(processes=self.threads)
            while neval:
                remaining = max(0, neval - self.threads)
                todo = neval - remaining
                neval -= todo
                args = ((learner, "learn_randomized", (experiment, )), ) * todo
                res = pool.map(unwrap_self_for_multiprocessing, args)
                ttt_scores = [ti for r in res for ti in r[0]]
                ttd_scores = [ti for r in res for ti in r[1]]
                ws.extend([r[2] for r in res])
        click.echo("Info: Finished learning.")

        # we only use weights from last iteration
        # ws = [ws[-1]]

        final_classifier = self.semi_supervised_learner.averaged_learner(ws)

        return final_classifier

    def _build_result(self, table, final_classifier, score_columns, experiment):

        weights = final_classifier.get_parameters()
        classifier_table = pd.DataFrame({'score': score_columns, 'weight': weights})

        scorer = Scorer(final_classifier, score_columns, experiment, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, self.tric_chromprob)

        scored_table = scorer.score(table)

        final_statistics, summary_statistics = scorer.get_error_stats()

        result = Result(summary_statistics, final_statistics, scored_table)

        click.echo("Info: Finished scoring and estimation statistics.")
        return result, scorer, classifier_table


@profile
def PyProphet(xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, tric_chromprob, threads, test):
    return HolyGostQuery(StandardSemiSupervisedLearner(LDALearner(), xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, test), ss_num_iter, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, tric_chromprob, threads, test)
