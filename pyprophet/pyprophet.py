from __future__ import division

import time
import operator
import multiprocessing
from collections import namedtuple
from contextlib import contextmanager
import pandas as pd
import numpy as np
import click

from .io._config import RunnerIOConfig, ErrorEstimationConfig
from .stats import (
    lookup_values_from_error_table,
    error_statistics,
    mean_and_std_dev,
    final_err_table,
    summary_err_table,
    posterior_chromatogram_hypotheses_fast,
)
from .data_handling import prepare_data_table, Experiment
from .classifiers import LDALearner, XGBLearner
from .semi_supervised import StandardSemiSupervisedLearner


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
        click.echo(
            "Info: Time needed for %s: %02d:%02d:%.1f" % (name, hours, minutes, needed)
        )
    else:
        click.echo("Info: Time needed: %02d:%02d:%.1f" % (hours, minutes, needed))


Result = namedtuple("Result", "summary_statistics final_statistics scored_tables")


def unwrap_self_for_multiprocessing(arg):
    """You can not call methods with multiprocessing, but free functions,
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

    def __init__(
        self,
        classifier,
        score_columns,
        experiment,
        group_id,
        error_estimation_config: ErrorEstimationConfig,
        tric_chromprob,
        ss_score_filter,
        color_palette,
        level,
    ):

        self.classifier = classifier
        self.score_columns = score_columns
        self.mu, self.nu = calculate_params_for_d_score(classifier, experiment)
        final_score = classifier.score(experiment, True)
        experiment["r_score"] = final_score
        experiment["d_score"] = (final_score - self.mu) / self.nu

        self.group_id = group_id
        self.error_estimation_config = error_estimation_config
        self.tric_chromprob = tric_chromprob
        self.ss_score_filter = ss_score_filter
        self.color_palette = color_palette
        self.level = level

        target_scores = experiment.get_top_target_peaks()["d_score"]
        decoy_scores = experiment.get_top_decoy_peaks()["d_score"]

        self.error_stat, self.pi0 = error_statistics(
            target_scores,
            decoy_scores,
            error_estimation_config.parametric,
            error_estimation_config.pfdr,
            error_estimation_config.pi0_lambda,
            error_estimation_config.pi0_method,
            error_estimation_config.pi0_smooth_df,
            error_estimation_config.pi0_smooth_log_pi0,
            True,  # compute_lfdr
            error_estimation_config.lfdr_truncate,
            error_estimation_config.lfdr_monotone,
            error_estimation_config.lfdr_transformation,
            error_estimation_config.lfdr_adj,
            error_estimation_config.lfdr_eps,
        )

        self.number_target_pg = len(experiment.df[experiment.df.is_decoy.eq(False)])
        self.number_target_peaks = len(experiment.get_top_target_peaks().df)
        self.dvals = experiment.df.loc[(experiment.df.is_decoy.eq(True)), "d_score"]
        self.target_scores = experiment.get_top_target_peaks().df["d_score"]
        self.decoy_scores = experiment.get_top_decoy_peaks().df["d_score"]

    def score(self, table):

        prepared_table, __ = prepare_data_table(
            table,
            self.ss_score_filter,
            tg_id_name=self.group_id,
            score_columns=self.score_columns,
            level=self.level,
        )
        texp = Experiment(prepared_table)
        score = self.classifier.score(texp, True)
        texp["r_score"] = score
        texp["d_score"] = (score - self.mu) / self.nu

        p_values, s_values, peps, q_values = lookup_values_from_error_table(
            texp["d_score"].values, self.error_stat
        )

        texp["pep"] = peps
        texp["q_value"] = q_values
        texp["s_value"] = s_values
        texp["p_value"] = p_values
        click.echo(
            "Info: Mean qvalue = %e, std_dev qvalue = %e"
            % (np.mean(q_values), np.std(q_values, ddof=1))
        )
        click.echo(
            "Info: Mean svalue = %e, std_dev svalue = %e"
            % (np.mean(s_values), np.std(s_values, ddof=1))
        )
        texp.add_peak_group_rank()

        df = table.join(
            texp[["r_score", "d_score", "p_value", "q_value", "pep", "peak_group_rank"]]
        )

        if self.tric_chromprob:
            df = self.add_chromatogram_probabilities(df, texp)

        return df

    def add_chromatogram_probabilities(self, scored_table, texp):
        allhypothesis, h0 = posterior_chromatogram_hypotheses_fast(
            texp, self.pi0["pi0"]
        )
        texp.df["h_score"] = allhypothesis
        texp.df["h0_score"] = h0
        scored_table = scored_table.join(texp[["h_score", "h0_score"]])

        return scored_table

    def get_error_stats(self):
        return final_err_table(self.error_stat), summary_err_table(self.error_stat)

    def minimal_error_stat(self):
        minimal_err_stat = ErrorStatistics(
            self.error_stat.df.loc[:, ["svalue", "qvalue", "pvalue", "pep", "cutoff"]],
            self.error_stat.num_null,
            self.error_stat.num_total,
        )
        return minimal_err_stat

    def __getstate__(self):
        """when pickling"""
        data = vars(self)
        data["error_stat"] = self.minimal_error_stat()
        return data

    def __setstate__(self, data):
        """when unpickling"""
        self.__dict__.update(data)


class PyProphet:
    """
    PyProphet assembles the semi-supervised scoring workflow, including training and applying
    statistical classifiers (e.g., LDA, XGBoost), estimating error rates, and generating scores.
    """

    def __init__(self, config: RunnerIOConfig):
        self.config = config
        self.rc = config.runner

        # Instantiate base learner
        if self.rc.classifier == "LDA":
            base_learner = LDALearner()
        elif self.rc.classifier == "XGBoost":
            base_learner = XGBLearner(
                self.rc.xgb_hyperparams,
                self.rc.xgb_params,
                self.rc.xgb_params_space,
                self.rc.threads,
            )
        else:
            raise click.ClickException(
                f"Classifier {self.rc.classifier} not supported."
            )

        # Build semi-supervised learner
        self.semi_supervised_learner = StandardSemiSupervisedLearner.from_config(
            config, base_learner
        )

    def _setup_experiment(self, table):
        prepared_table, score_columns = prepare_data_table(
            table,
            self.rc.ss_score_filter,
            tg_id_name=self.rc.group_id,
            level=self.config.level,
        )
        experiment = Experiment(prepared_table)
        experiment.log_summary()
        return experiment, score_columns

    def apply_weights(self, table, loaded_weights):
        with timer("Apply Weights"):
            experiment, score_columns = self._setup_experiment(table)

            if self.rc.classifier == "LDA":
                if np.all(score_columns == loaded_weights["score"].values):
                    weights = loaded_weights["weight"].values
                else:
                    raise click.ClickException(
                        "Scores in weights file do not match data."
                    )
            elif self.rc.classifier == "XGBoost":
                weights = loaded_weights

            final_classifier = self._apply_weights_on_exp(experiment, weights)
            return self._build_result(
                table, final_classifier, score_columns, experiment
            )

    def _apply_weights_on_exp(self, experiment, weights):
        learner = self.semi_supervised_learner
        click.echo("Info: Applying pretrained weights.")

        clf_scores = learner.score(experiment, weights)
        experiment.set_and_rerank("classifier_score", clf_scores)

        if self.rc.classifier == "LDA":
            ws = [weights.flatten()]
            return learner.averaged_learner(ws)
        else:  # XGBoost
            return learner.set_learner(weights)

    @profile
    def learn_and_apply(self, table):
        with timer("Learn and Apply"):
            experiment, score_columns = self._setup_experiment(table)
            final_classifier = self._learn(experiment, score_columns)
            return self._build_result(
                table, final_classifier, score_columns, experiment
            )

    def _learn(self, experiment, score_columns):
        learner = self.semi_supervised_learner
        neval = self.rc.ss_num_iter
        ttt, ttd, ws = [], [], []

        click.echo(f"Info: Learning on {neval} folds with {self.rc.threads} threads.")

        if self.rc.threads == 1:
            for _ in range(neval):
                ttt_scores, ttd_scores, w = learner.learn_randomized(
                    experiment, score_columns, 1
                )
                ttt.append(ttt_scores)
                ttd.append(ttd_scores)
                ws.append(w)
        else:
            pool = multiprocessing.Pool(processes=self.rc.threads)
            while neval:
                todo = min(neval, self.rc.threads)
                args = tuple(
                    (learner, "learn_randomized", (experiment, score_columns, tid))
                    for tid in range(1, todo + 1)
                )
                res = pool.map(unwrap_self_for_multiprocessing, args)
                ttt += [r[0] for r in res]
                ttd += [r[1] for r in res]
                ws += [r[2] for r in res]
                neval -= todo

        if self.rc.classifier == "LDA":
            return learner.averaged_learner(ws)

        # XGBoost - integrate cross-validation scores and train final model
        ttt_avg = pd.concat(ttt, axis=1).mean(axis=1)
        ttd_avg = pd.concat(ttd, axis=1).mean(axis=1)
        integrated_scores = pd.concat([ttt_avg, ttd_avg], axis=0)
        experiment.set_and_rerank("classifier_score", integrated_scores)

        model = learner.learn_final(experiment)
        return learner.set_learner(model)

    def _build_result(self, table, final_classifier, score_columns, experiment):
        # Collect weights
        if self.rc.classifier == "LDA":
            weights = final_classifier.get_parameters()
            classifier_table = pd.DataFrame({"score": score_columns, "weight": weights})
        else:
            classifier_table = final_classifier.get_parameters()
            mapped = {
                f: v
                for f, v in zip(
                    score_columns, classifier_table.get("importance", {}).values()
                )
            }
            for feat, importance in sorted(
                mapped.items(), key=operator.itemgetter(1), reverse=True
            ):
                click.echo(f"Info: Importance of {feat}: {importance}")

        # Score the table
        scorer = Scorer(
            classifier=final_classifier,
            score_columns=score_columns,
            experiment=experiment,
            group_id=self.rc.group_id,
            error_estimation_config=self.rc.error_estimation_config,
            tric_chromprob=self.rc.tric_chromprob,
            ss_score_filter=self.rc.ss_score_filter,
            color_palette=self.rc.color_palette,
            level=self.config.level,
        )

        scored_table = scorer.score(table)
        final_stats, summary_stats = scorer.get_error_stats()
        result = Result(summary_stats, final_stats, scored_table)

        click.echo("Info: Scoring and statistics complete.")
        return result, scorer, classifier_table
