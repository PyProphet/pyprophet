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

# set output options for regression tests on a wide terminal
pd.set_option('display.width', 180)

# reduce precision to avoid to sensitive tests because of roundings:
pd.set_option('display.precision', 6)

from stats import (lookup_s_and_q_values_from_error_table, calculate_final_statistics,
                   mean_and_std_dev, final_err_table, summary_err_table,
                   posterior_pg_prob, posterior_chromatogram_hypotheses_fast,
                   ErrorStatistics)

from config import CONFIG

from data_handling import (prepare_data_tables, prepare_data_table, Experiment, check_header,
                           sample_data_tables, read_csv)
from classifiers import (LDALearner)
from semi_supervised import (AbstractSemiSupervisedLearner, StandardSemiSupervisedLearner)

import multiprocessing

from std_logger import logging

import time

from collections import namedtuple
from contextlib import contextmanager


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


def unwrap_self_for_multiprocessing((inst, method_name, args),):
    """ You can not call methods with multiprocessing, but free functions,
        If you want to call  inst.method(arg0, arg1),

            unwrap_self_for_multiprocessing(inst, "method", (arg0, arg1))

        does the trick.
    """
    return getattr(inst, method_name)(*args)


@profile
def calculate_params_for_d_score(classifier, experiment):
    score = classifier.score(experiment, True)
    experiment.set_and_rerank("classifier_score", score)

    td_scores = experiment.get_top_decoy_peaks()["classifier_score"]

    mu, nu = mean_and_std_dev(td_scores)
    return mu, nu


class Scorer(object):

    def __init__(self, classifier, score_columns, experiment, all_test_target_scores,
                 all_test_decoy_scores, merge_results):

        self.classifier = classifier
        self.score_columns = score_columns
        self.mu, self.nu = calculate_params_for_d_score(classifier, experiment)
        self.merge_results = merge_results
        final_score = classifier.score(experiment, True)
        experiment["d_score"] = (final_score - self.mu) / self.nu
        lambda_ = CONFIG.get("final_statistics.lambda")

        all_tt_scores = experiment.get_top_target_peaks()["d_score"]

        use_pemp = CONFIG.get("final_statistics.emp_p")
        use_pfdr = CONFIG.get("final_statistics.pfdr")

        self.error_stat, self.target_pvalues = calculate_final_statistics(all_tt_scores,
                                                                          all_test_target_scores,
                                                                          all_test_decoy_scores,
                                                                          lambda_,
                                                                          use_pemp,
                                                                          use_pfdr)

        self.number_target_pg = len(experiment.df[experiment.df.is_decoy.eq(False)])
        self.number_target_peaks = len(experiment.get_top_target_peaks().df)
        self.dvals = experiment.df.loc[(experiment.df.is_decoy.eq(True)), "d_score"]
        self.target_scores = experiment.get_top_target_peaks().df["d_score"]
        self.decoy_scores = experiment.get_top_decoy_peaks().df["d_score"]

    def score(self, table):

        prepared_table, __ = prepare_data_table(table, score_columns=self.score_columns)
        texp = Experiment(prepared_table)
        score = self.classifier.score(texp, True)
        texp["d_score"] = (score - self.mu) / self.nu

        s_values, q_values = lookup_s_and_q_values_from_error_table(texp["d_score"].values,
                                                                    self.error_stat.df)
        texp["m_score"] = q_values
        texp["s_value"] = s_values
        logging.info("mean m_score = %e, std_dev m_score = %e" % (np.mean(q_values),
                                                                  np.std(q_values, ddof=1)))
        logging.info("mean s_value = %e, std_dev s_value = %e" % (np.mean(s_values),
                                                                  np.std(s_values, ddof=1)))
        texp.add_peak_group_rank()

        df = table.join(texp[["d_score", "m_score", "peak_group_rank"]])

        if CONFIG.get("compute.probabilities"):
            df = self.add_probabilities(df, texp)

        if CONFIG.get("target.compress_results"):
            to_drop = [n for n in df.columns if n.startswith("var_") or n.startswith("main_")]
            df.drop(to_drop, axis=1, inplace=True)

        return df

    def add_probabilities(self, scored_table, texp):

        lambda_ = CONFIG.get("final_statistics.lambda")
        pp_pg_pvalues = posterior_pg_prob(self.dvals, self.target_scores, self.decoy_scores,
                                          self.error_stat, self.number_target_peaks,
                                          self.number_target_pg,
                                          texp.df["d_score"],
                                          lambda_)
        texp.df["pg_score"] = pp_pg_pvalues
        scored_table = scored_table.join(texp[["pg_score"]])

        prior_chrom_null = self.error_stat.num_null / self.error_stat.num_total
        allhypothesis, h0 = posterior_chromatogram_hypotheses_fast(texp, prior_chrom_null)
        texp.df["h_score"] = allhypothesis
        texp.df["h0_score"] = h0
        scored_table = scored_table.join(texp[["h_score", "h0_score"]])

        return scored_table

    def score_many(self, tables):
        if self.merge_results:
            df = pd.concat([self.score(t) for t in tables])
            yield ScoredTable(df)
        else:
            for table in tables:
                df = self.score(table)
                yield ScoredTable(df)

    def score_many_lazy(self, pathes, delim_in):
        return LazyScoredTablesIter(pathes, self.merge_results, self, delim_in=delim_in)

    """
    TODO: out of core weight applier und alten weight applier von commandline aufrufen
    und unterschiede finden !!!!
    """

    def get_error_stats(self):
        return final_err_table(self.error_stat.df), summary_err_table(self.error_stat.df)

    def minimal_error_stat(self):
        minimal_err_stat = ErrorStatistics(self.error_stat.df.loc[:, ["svalue", "qvalue", "cutoff"]],
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


class LazyScoredTablesIter(object):

    def __init__(self, pathes, merge_results, scorer, **options):
        self.pathes = pathes
        self.merge_results = merge_results
        self.scorer = scorer
        self.options = options

    def __iter__(self):
        if self.merge_results:
            yield MergedLazyScoredTables(self.pathes, self.scorer, self.options)

        else:
            for p in self.pathes:
                yield LazyScoredTable(p, self.scorer, self.options)


class ScoredTable(object):

    def __init__(self, df):
        self.df = df

    def to_csv(self, out_path, out_path_filtered, cutoff, sep,  **kw):
        df = self.df
        df.to_csv(out_path, sep, **kw)
        df = df[df.d_score > cutoff]
        df.to_csv(out_path_filtered, sep, **kw)

    def scores(self):
        decoys = self.df[self.df["decoy"] == 1]["d_score"].values
        targets = self.df[self.df["decoy"] == 0]["d_score"].values

        tops = self.df[self.df["peak_group_rank"] == 1]
        top_decoys = tops[tops["decoy"] == 1]["d_score"].values
        top_targets = tops[tops["decoy"] == 0]["d_score"].values

        return decoys, targets, top_decoys, top_targets


class LazyScoredTable(object):

    def __init__(self, path, scorer, options):
        self.path = path
        self.scorer = scorer
        self.options = options
        self.decoys = self.targets = self.top_decoys = self.top_targets = None

    def to_csv(self, out_path, out_path_filtered, cutoff, sep, **kw):
        table = read_csv(self.path, self.options.get("delim_in"))
        df = self.scorer.score(table)
        df.to_csv(out_path, sep, **kw)
        df = df[df.d_score > cutoff]
        df.to_csv(out_path_filtered, sep, **kw)

        self.decoys = df[df["decoy"] == 1]["d_score"].values
        self.targets = df[df["decoy"] == 0]["d_score"].values

        tops = df[df["peak_group_rank"] == 1]
        self.top_decoys = tops[tops["decoy"] == 1]["d_score"].values
        self.top_targets = tops[tops["decoy"] == 0]["d_score"].values

    def scores(self):
        assert self.decoys is not None, ("you have to save the lazy table before you can access "
                                         "scores")
        return self.decoys, self.targets, self.top_decoys, self.top_targets


class MergedLazyScoredTables(object):

    def __init__(self, pathes, scorer, options):
        self.pathes = pathes
        self.scorer = scorer
        self.options = options
        self.decoys = self.targets = self.top_decoys = self.top_targets = None

    def to_csv(self, out_path, out_path_filtered, cutoff, sep, **kw):
        # write first table with header
        path = self.pathes[0]
        table = read_csv(path, self.options.get("delim_in"))

        df = self.scorer.score(table)
        df.to_csv(out_path, sep, header=True, **kw)
        self._update_scores(df)
        df = df[df.d_score > cutoff]
        df.to_csv(out_path_filtered, sep, header=True, **kw)

        # now append and do not write headers again:
        for path in self.pathes[1:]:
            with open(out_path, "a") as fp, open(out_path_filtered, "a") as fp2:
                table = read_csv(path, self.options.get("delim_in"))

                df = self.scorer.score(table)
                df.to_csv(fp, sep, header=True, **kw)
                self._update_scores(df)
                df = df[df.d_score > cutoff]
                df.to_csv(fp2, sep, header=True, **kw)

    def scores(self):
        assert self.decoys is not None, ("you have to save the lazy table before you can access "
                                         "scores")
        return self.decoys, self.targets, self.top_decoys, self.top_targets

    def _update_scores(self, scored_table):

        if self.decoys is None:
            self.decoys = self.targets = self.top_decoys = self.top_targets = [] * 4

        decoys = scored_table[scored_table["decoy"] == 1]["d_score"].values
        targets = scored_table[scored_table["decoy"] == 0]["d_score"].values

        tops = scored_table[scored_table["peak_group_rank"] == 1]
        top_decoys = tops[tops["decoy"] == 1]["d_score"].values
        top_targets = tops[tops["decoy"] == 0]["d_score"].values

        self.decoys.extend(decoys)
        self.targets.extend(targets)
        self.top_decoys.extend(top_decoys)
        self.top_targets.extend(top_targets)


class HolyGostQuery(object):

    """ HolyGhostQuery assembles the unsupervised methods.
        See below how PyProphet parameterises this class.
    """

    def __init__(self, semi_supervised_learner):
        assert isinstance(semi_supervised_learner,
                          AbstractSemiSupervisedLearner)
        self.semi_supervised_learner = semi_supervised_learner

    def read_tables_iter(self, pathes, delim):
        logging.info("process %s" % ", ".join(pathes))
        for path in pathes:
            part = read_csv(path, delim)
            yield part

    def check_table_headers(self, pathes, delim, check_cols):
        headers = set()
        for path in pathes:
            header = check_header(path, delim, check_cols)
            headers.add(tuple(header))
        if len(headers) > 1:
            raise Exception("the input files have different headers.")

    def _setup_experiment(self, tables):
        prepared_tables, score_columns = prepare_data_tables(tables)
        prepared_table = pd.concat(prepared_tables)
        experiment = Experiment(prepared_table)
        experiment.log_summary()
        return experiment, score_columns

    def apply_weights(self, pathes, delim_in, check_cols, loaded_weights):

        self.check_table_headers(pathes, delim_in, check_cols)
        tables = list(self.read_tables_iter(pathes, delim_in))
        with timer():
            logging.info("apply weights")
            result, scorer, trained_weights = self._apply_weights(tables, loaded_weights)
            logging.info("processing input data finished")
        return result, scorer, trained_weights

    def _apply_weights(self, tables, loaded_weights):

        experiment, score_columns = self._setup_experiment(tables)

        final_classifier, all_test_target_scores, all_test_decoy_scores = \
            self._apply_weights_on_exp(experiment, loaded_weights)

        return self._build_result(tables, final_classifier, score_columns, experiment,
                                  all_test_target_scores, all_test_decoy_scores)

    def _apply_weights_on_exp(self, experiment, loaded_weights):

        learner = self.semi_supervised_learner

        logging.info("start application of pretrained weights")
        clf_scores = learner.score(experiment, loaded_weights)
        experiment.set_and_rerank("classifier_score", clf_scores)

        all_test_target_scores = experiment.get_top_target_peaks()["classifier_score"]
        all_test_decoy_scores = experiment.get_top_decoy_peaks()["classifier_score"]
        logging.info("finished pretrained scoring")

        ws = [loaded_weights.flatten()]
        final_classifier = self.semi_supervised_learner.averaged_learner(ws)

        return final_classifier, all_test_target_scores, all_test_decoy_scores

    def apply_weights_out_of_core(self, pathes, delim, check_cols, loaded_weights):
        self.check_table_headers(pathes, delim, check_cols)
        with timer():

            logging.info("apply weights out of core")
            result, scorer, trained_weights = self._apply_weights_out_of_core(pathes, delim,
                                                                              loaded_weights)
            logging.info("processing input data finished")

        return result, scorer, trained_weights

    def _apply_weights_out_of_core(self, pathes, delim, loaded_weights):

        sampling_rate = CONFIG.get("out_of_core.sampling_rate")
        assert 0 < sampling_rate <= 1.0, "invalid sampling rate value"
        prepared_tables, score_columns = sample_data_tables(pathes, delim, sampling_rate)
        prepared_table = pd.concat(prepared_tables)

        experiment = Experiment(prepared_table)
        experiment.log_summary()

        final_classifier, all_test_target_scores, all_test_decoy_scores = \
            self._apply_weights_on_exp(experiment, loaded_weights)

        return self._build_lazy_result(pathes, final_classifier, score_columns, experiment,
                                       all_test_target_scores, all_test_decoy_scores)

    @profile
    def apply_scorer_out_of_core(self, pathes, delim, check_cols, loaded_scorer):
        self.check_table_headers(pathes, delim, check_cols)
        with timer():
            logging.info("apply scorer to input data")
            result, __, used_weights = self._apply_scorer_out_of_core(pathes, delim, loaded_scorer)
            logging.info("processing input data finished")
        return result, None, used_weights

    @profile
    def _apply_scorer_out_of_core(self, pathes, delim, scorer):

        merge_results = CONFIG.get("multiple_files.merge_results")
        # TODO: merge_resuls has nothing to do with scorer, we need extra class for
        # writing results, maybe lazy....:
        scorer.merge_results = merge_results
        delim_in = CONFIG.get("delim.in")
        scored_tables_lazy = scorer.score_many_lazy(pathes, delim_in)
        final_statistics, summary_statistics = scorer.get_error_stats()
        weights = scorer.classifier.get_parameters()
        return Result(None, None, scored_tables_lazy), None, weights

    @profile
    def apply_scorer(self, pathes, delim, check_cols, loaded_scorer):
        self.check_table_headers(pathes, delim, check_cols)
        tables = list(self.read_tables_iter(pathes, delim))

        with timer():
            logging.info("apply scorer to input data")
            result, __, trained_weights = self._apply_scorer(tables, loaded_scorer)
            scorer = None
            logging.info("processing input data finished")
        return result, scorer, trained_weights

    @profile
    def _apply_scorer(self, tables, loaded_scorer):
        scored_tables = loaded_scorer.score_many(tables)
        trained_weights = loaded_scorer.classifier.get_parameters()
        return Result(None, None, scored_tables), None, trained_weights

    @profile
    def _learn_and_apply_out_of_core(self, pathes, delim):

        sampling_rate = CONFIG.get("out_of_core.sampling_rate")
        assert 0 < sampling_rate <= 1.0, "invalid sampling rate value"
        prepared_tables, score_columns = sample_data_tables(pathes, delim, sampling_rate)
        prepared_table = pd.concat(prepared_tables)
        experiment = Experiment(prepared_table)
        experiment.log_summary()
        final_classifier, all_test_target_scores, all_test_decoy_scores = self._learn(experiment)
        return self._build_lazy_result(pathes, final_classifier, score_columns, experiment,
                                       all_test_target_scores, all_test_decoy_scores)

    @profile
    def learn_and_apply_out_of_core(self, pathes, delim, check_cols):

        self.check_table_headers(pathes, delim, check_cols)
        with timer():

            logging.info("learn and apply classifier out of core")
            result, scorer, trained_weights = self._learn_and_apply_out_of_core(pathes, delim)
            logging.info("processing input data finished")

        return result, scorer, trained_weights

    @profile
    def learn_and_apply(self, pathes, delim, check_cols):

        self.check_table_headers(pathes, delim, check_cols)
        tables = list(self.read_tables_iter(pathes, delim))
        with timer():

            logging.info("learn and apply classifier from input data")
            result, scorer, trained_weights = self._learn_and_apply(tables)
            logging.info("processing input data finished")

        return result, scorer, trained_weights

    def _learn_and_apply(self, tables):

        experiment, score_columns = self._setup_experiment(tables)
        final_classifier, all_test_target_scores, all_test_decoy_scores = self._learn(experiment)

        return self._build_result(tables, final_classifier, score_columns, experiment,
                                  all_test_target_scores, all_test_decoy_scores)

    def _learn(self, experiment):
        is_test = CONFIG.get("is_test")
        if is_test:  # for reliable results
            experiment.df.sort("tg_id", ascending=True, inplace=True)

        learner = self.semi_supervised_learner
        ws = []

        neval = CONFIG.get("xeval.num_iter")
        num_processes = CONFIG.get("num_processes")
        all_test_target_scores = []
        all_test_decoy_scores = []

        logging.info("learn and apply scorer")
        logging.info("start %d cross evals using %d processes" % (neval, num_processes))

        if num_processes == 1:
            for k in range(neval):
                (ttt_scores, ttd_scores, w) = learner.learn_randomized(experiment)
                all_test_target_scores.extend(ttt_scores)
                all_test_decoy_scores.extend(ttd_scores)
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
                all_test_target_scores.extend(ttt_scores)
                all_test_decoy_scores.extend(ttd_scores)
        logging.info("finished cross evals")
        logging.info("")

        # only use socres from last iteration to build statistical model:
        if CONFIG.get("semi_supervised_learner.stat_best"):
            all_test_target_scores = ttt_scores
            all_test_decoy_scores = ttd_scores

        # we only use weights from last iteration if indicated:
        if CONFIG.get("semi_supervised_learner.use_best"):
            ws = [ws[-1]]

        final_classifier = self.semi_supervised_learner.averaged_learner(ws)

        return final_classifier, all_test_target_scores, all_test_decoy_scores

    def _build_result(self, tables, final_classifier, score_columns, experiment,
                      all_test_target_scores, all_test_decoy_scores):

        merge_results = CONFIG.get("multiple_files.merge_results")
        weights = final_classifier.get_parameters()
        scorer = Scorer(final_classifier, score_columns, experiment, all_test_target_scores,
                        all_test_decoy_scores, merge_results)

        scored_tables = list(scorer.score_many(tables))

        final_statistics, summary_statistics = scorer.get_error_stats()

        result = Result(summary_statistics, final_statistics, scored_tables)

        logging.info("calculated scoring and statistics")
        return result, scorer, weights

    def _build_lazy_result(self, pathes, final_classifier, score_columns, experiment,
                           all_test_target_scores, all_test_decoy_scores):

        merge_results = CONFIG.get("multiple_files.merge_results")
        weights = final_classifier.get_parameters()
        scorer = Scorer(final_classifier, score_columns, experiment, all_test_target_scores,
                        all_test_decoy_scores, merge_results)

        delim_in = CONFIG.get("delim.in")
        scored_tables_lazy = scorer.score_many_lazy(pathes, delim_in)

        final_statistics, summary_statistics = scorer.get_error_stats()

        result = Result(summary_statistics, final_statistics, scored_tables_lazy)

        logging.info("calculated scoring and statistics")
        return result, scorer, weights


@profile
def PyProphet():
    return HolyGostQuery(StandardSemiSupervisedLearner(LDALearner()))
