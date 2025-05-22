import abc
import click
import sys
import os
import time
import warnings
from loguru import logger
import pandas as pd
import numpy as np
import polars as pl
import sqlite3
import duckdb
import pickle


from ._config import RunnerIOConfig
from .io.util import setup_logger, check_sqlite_table
from .io.dispatcher import ReaderDispatcher, WriterDispatcher
from .pyprophet import PyProphet
from .glyco.scoring import partial_score, combined_score
from .glyco.stats import ErrorStatisticsCalculator

try:
    profile
except NameError:

    def profile(fun):
        return fun


setup_logger()


class PyProphetRunner(object):

    __metaclass__ = abc.ABCMeta

    """Base class for workflow of command line tool
    """

    def __init__(
        self,
        config: RunnerIOConfig,
    ):
        self.config = config
        self.reader = ReaderDispatcher.get_reader(config)
        self.writer = WriterDispatcher.get_writer(config)
        self.table = self.reader.read()

    @property
    def classifier(self):
        return self.config.runner.classifier

    @property
    def infile(self):
        return self.config.infile

    @property
    def outfile(self):
        return self.config.outfile

    @property
    def level(self):
        return self.config.level

    @property
    def glyco(self):
        return self.config.runner.glyco

    @property
    def runner_config(self):
        return self.config.runner

    @property
    def error_estimation_config(self):
        return self.config.runner.error_estimation_config

    @abc.abstractmethod
    def run_algo(self, part=None):
        pass

    def run(self):

        self.check_cols = [self.runner_config.group_id, "run_id", "decoy"]

        if self.glyco and self.level in ["ms2", "ms1ms2"]:
            start_at = time.time()

            start_at_peptide = time.time()
            click.echo("*" * 30 + "  Glycoform Scoring  " + "*" * 30)
            click.echo("-" * 80)
            click.echo("Info: Scoring peptide part")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (result_peptide, _scorer_peptide, weights_peptide) = self.run_algo(
                    part="peptide"
                )
            end_at_peptide = time.time() - start_at_peptide
            seconds = int(end_at_peptide)
            msecs = int(1000 * (end_at_peptide - seconds))
            click.echo(
                "Info: peptide part scored: %d seconds and %d msecs" % (seconds, msecs)
            )

            start_at_glycan = time.time()
            click.echo("-" * 80)
            click.echo("Info: Scoring glycan part")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (result_glycan, scorer_glycan, weights_glycan) = partial_score(
                    self, part="glycan"
                )

            end_at_glycan = time.time() - start_at_glycan
            seconds = int(end_at_glycan)
            msecs = int(1000 * (end_at_glycan - seconds))
            click.echo(
                "Info: glycan part scored: %d seconds and %d msecs" % (seconds, msecs)
            )

            start_at_combined = time.time()
            click.echo("-" * 80)
            click.echo("Info: Calculating combined scores")
            (result_combined, weights_combined) = combined_score(
                self.runner_config.group_id, result_peptide, result_glycan
            )

            if isinstance(weights_combined, pd.DataFrame):
                click.echo(weights_combined)

            end_at_combined = time.time() - start_at_combined
            seconds = int(end_at_combined)
            msecs = int(1000 * (end_at_combined - seconds))
            click.echo(
                "Info: combined scores calculated: %d seconds and %d msecs"
                % (seconds, msecs)
            )

            start_at_stats = time.time()
            click.echo("-" * 80)
            click.echo("Info: Calculating error statistics")
            error_stat = ErrorStatisticsCalculator(
                result_combined,
                density_estimator=self.runner_config.density_estimator,
                grid_size=self.runner_config.grid_size,
                parametric=self.error_estimation_config.parametric,
                pfdr=self.error_estimation_config.pfdr,
                pi0_lambda=self.error_estimation_config.pi0_lambda,
                pi0_method=self.error_estimation_config.pi0_method,
                pi0_smooth_df=self.error_estimation_config.pi0_smooth_df,
                pi0_smooth_log_pi0=self.error_estimation_config.pi0_smooth_log_pi0,
                lfdr_truncate=self.error_estimation_config.lfdr_truncate,
                lfdr_monotone=self.error_estimation_config.lfdr_monotone,
                lfdr_transformation=self.error_estimation_config.lfdr_transformation,
                lfdr_adj=self.error_estimation_config.lfdr_adj,
                lfdr_eps=self.error_estimation_config.lfdr_eps,
                tric_chromprob=self.runner_config.tric_chromprob,
            )
            result, pi0 = error_stat.error_statistics()

            end_at_stats = time.time() - start_at_stats
            seconds = int(end_at_stats)
            msecs = int(1000 * (end_at_stats - seconds))
            click.echo(
                "Info: error statistics finished: %d seconds and %d msecs"
                % (seconds, msecs)
            )

            if all(
                (
                    isinstance(w, pd.DataFrame)
                    for w in [weights_peptide, weights_glycan, weights_combined]
                )
            ):
                weights = pd.concat(
                    (
                        weights_peptide.assign(part="peptide"),
                        weights_glycan.assign(part="glycan"),
                        weights_combined.assign(part="combined"),
                    ),
                    ignore_index=True,
                )
            else:
                weights = {
                    "peptide": weights_peptide,
                    "glycan": weights_glycan,
                    "combined": weights_combined,
                }

            needed = time.time() - start_at
        else:
            start_at = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (result, scorer, weights) = self.run_algo()
            needed = time.time() - start_at

        self.print_summary(result)

        if self.glyco and self.level in ["ms2", "ms1ms2"]:
            self.writer.save_results(result_peptide, pi0)
        else:
            if self.config.subsample_ratio == 1:
                # We don't want to save the results if we subsampled the data, the second pass of applying the weights to the full data is when we save the results
                self.writer.save_results(result, scorer.pi0)
        self.writer.save_weights(weights)

        seconds = int(needed)
        msecs = int(1000 * (needed - seconds))

        logger.info("Total time: %d seconds and %d msecs wall time" % (seconds, msecs))

        return self.config.extra_writes.get("trained_weights_path")

    def print_summary(self, result):
        if result.summary_statistics is not None:
            click.echo("=" * 80)
            click.echo(result.summary_statistics)
            click.echo("=" * 80)


class PyProphetLearner(PyProphetRunner):

    def run_algo(self, part=None):
        if self.glyco:
            if (
                self.level in ["ms2", "ms1ms2"]
                and part != "peptide"
                and part != "glycan"
            ):
                raise click.ClickException(
                    "For glycopeptide MS2-level scoring, please specify either 'peptide' or 'glycan' as part."
                )

            if "decoy" in self.table.columns and self.level != "transition":
                self.table = self.table.drop(columns=["decoy"])
            if self.level == "ms2" or self.level == "ms1ms2":
                self.table = self.table.rename(columns={"decoy_" + part: "decoy"})
            elif self.level == "ms1":
                self.table = self.table.rename(columns={"decoy_glycan": "decoy"})

        (result, scorer, weights) = PyProphet(self.config).learn_and_apply(self.table)
        return (result, scorer, weights)


class PyProphetWeightApplier(PyProphetRunner):

    def __init__(self, apply_weights: str, config: RunnerIOConfig):
        super(PyProphetWeightApplier, self).__init__(config)

        if not os.path.exists(apply_weights):
            raise click.ClickException(
                "Weights file %s does not exist." % apply_weights
            )
        if self.config.file_type in (
            "tsv",
            "parquet",
            "parquet_split",
            "parquet_split_multi",
        ):
            if self.classifier == "LDA":
                try:
                    self.persisted_weights = pd.read_csv(apply_weights, sep=",")
                    if self.level != self.persisted_weights["level"].unique()[0]:
                        raise click.ClickException("Weights file has wrong level.")
                except Exception:
                    import traceback

                    traceback.print_exc()
                    raise
            elif self.classifier == "XGBoost":
                with open(apply_weights, "rb") as file:
                    self.persisted_weights = pickle.load(file)
        elif self.config.file_type == "osw":
            if self.classifier == "LDA":
                try:
                    con = sqlite3.connect(apply_weights)

                    if not check_sqlite_table(con, "PYPROPHET_WEIGHTS"):
                        raise click.ClickException(
                            "PYPROPHET_WEIGHTS table is not present in file, cannot apply weights for LDA classifier! Make sure you have run the scoring on a subset of the data first, or that you supplied the right `--classifier` parameter."
                        )
                    data = pd.read_sql_query(
                        "SELECT * FROM PYPROPHET_WEIGHTS WHERE LEVEL=='%s'"
                        % self.level,
                        con,
                    )
                    data.columns = [col.lower() for col in data.columns]
                    con.close()
                    self.persisted_weights = data
                    if self.level != self.persisted_weights["level"].unique()[0]:
                        raise click.ClickException("Weights file has wrong level.")
                except Exception:
                    import traceback

                    traceback.print_exc()
                    raise
            elif self.classifier == "XGBoost":
                try:
                    con = sqlite3.connect(apply_weights)

                    if not check_sqlite_table(con, "PYPROPHET_XGB"):
                        raise click.ClickException(
                            "PYPROPHET_XGB table is not present in file, cannot apply weights for XGBoost classifier! Make sure you have run the scoring on a subset of the data first, or that you supplied the right `--classifier` parameter."
                        )
                    data = con.execute(
                        "SELECT xgb FROM PYPROPHET_XGB WHERE LEVEL=='%s'" % self.level
                    ).fetchone()
                    con.close()
                    self.persisted_weights = pickle.loads(data[0])
                except Exception:
                    import traceback

                    traceback.print_exc()
                    raise

    def run_algo(self):
        (result, scorer, weights) = PyProphet(self.config).apply_weights(
            self.table, self.persisted_weights
        )
        return (result, scorer, weights)
