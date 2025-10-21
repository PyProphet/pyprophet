"""
This module defines the core classes and workflows for running PyProphet, a tool for
statistical scoring and error estimation in targeted proteomics and glycoproteomics data analysis.

Classes:
    - PyProphetRunner: Base class for running PyProphet workflows.
    - PyProphetLearner: Implements the learning and scoring workflow.
    - PyProphetWeightApplier: Applies pre-trained weights to new datasets.

Functions:
    - profile: A no-op decorator for profiling (used if no profiler is available).
"""

import abc
import os
import pickle
import sqlite3
import time
import warnings

import click
import pandas as pd
from loguru import logger

from .._config import RunnerIOConfig
from ..glyco.scoring import combined_score, partial_score
from ..glyco.stats import ErrorStatisticsCalculator
from ..io.dispatcher import ReaderDispatcher, WriterDispatcher
from ..io.util import check_sqlite_table
from .pyprophet import PyProphet

try:
    profile
except NameError:

    def profile(fun):
        return fun


class PyProphetRunner(object):
    """
    Base class for running PyProphet workflows.

    This class provides the core structure for executing PyProphet workflows, including
    reading input data, running algorithms, and saving results.

    Attributes:
        config (RunnerIOConfig): Configuration object for the workflow.
        reader (BaseReader): Reader object for input data.
        writer (BaseWriter): Writer object for output data.
        table (pd.DataFrame): The input data table.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        config: RunnerIOConfig,
    ):
        self.config = config
        self.reader = ReaderDispatcher.get_reader(config)
        self.writer = WriterDispatcher.get_writer(config)
        logger.debug(
            f"Using reader: {self.reader.__class__.__name__} for file type: {self.config.file_type}"
        )
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
        """
        Abstract method for running the algorithm.

        Args:
            part (str, optional): Specifies the part of the workflow to run (e.g., "peptide", "glycan"). Specific for glycopeptide workflows.
        """
        pass

    def run(self):
        """
        Executes the PyProphet workflow, including scoring, error estimation, and saving results.
        """
        if self.classifier == "HistGradientBoosting":
            # We need to adjust the parallelism used throughout scoring to avoid oversubscription, since HistGradientBoosting uses multiple threads internally
            total_threads = int(os.getenv("TOTAL_CPUS", os.cpu_count()))
            # Assert threads is not greater than total_threads
            assert self.runner_config.threads <= total_threads, (
                f"You requested {self.runner_config.threads} threads, but only {total_threads} are available."
            )
            
            # Log the OMP_NUM_THREADS value that's currently set
            omp_value = os.getenv("OMP_NUM_THREADS", "NOT SET")
            logger.info(
                f"HistGradientBoosting parallelism: "
                f"semi-supervised threads={self.runner_config.threads}, "
                f"OMP_NUM_THREADS={omp_value}, "
                f"total_cpus={total_threads}"
            )
            
            # Note: OMP_NUM_THREADS should ideally be set BEFORE pyprophet is launched
            # The CLI now sets it automatically in main.py before any numpy imports
            if "OMP_NUM_THREADS" not in os.environ:
                logger.warning(
                    "OMP_NUM_THREADS was not set. This should have been set automatically by main.py. "
                    "Something may be wrong with the initialization sequence."
                )
            elif os.getenv("_PYPROPHET_OMP_AUTO") == "1":
                logger.info(
                    "OMP_NUM_THREADS was automatically set by PyProphet. "
                    "For explicit control, set it before launching pyprophet."
                )
            else:
                logger.info(
                    "OMP_NUM_THREADS was set externally (recommended for best control)."
                )

        self.check_cols = [self.runner_config.group_id, "run_id", "decoy"]

        if self.glyco and self.level in ["ms2", "ms1ms2"]:
            start_at = time.time()

            start_at_peptide = time.time()
            logger.opt(raw=True).info("*" * 30 + "  Glycoform Scoring  " + "*" * 30)
            logger.opt(raw=True).info("-" * 80)
            logger.info("Scoring peptide part")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (result_peptide, _scorer_peptide, weights_peptide) = self.run_algo(
                    part="peptide"
                )
            end_at_peptide = time.time() - start_at_peptide
            seconds = int(end_at_peptide)
            msecs = int(1000 * (end_at_peptide - seconds))
            logger.info(
                "peptide part scored: %d seconds and %d msecs" % (seconds, msecs)
            )

            start_at_glycan = time.time()
            logger.opt(raw=True).info("-" * 80)
            logger.info("Scoring glycan part")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (result_glycan, scorer_glycan, weights_glycan) = partial_score(
                    self, part="glycan"
                )

            end_at_glycan = time.time() - start_at_glycan
            seconds = int(end_at_glycan)
            msecs = int(1000 * (end_at_glycan - seconds))
            logger.info(
                "glycan part scored: %d seconds and %d msecs" % (seconds, msecs)
            )

            start_at_combined = time.time()
            logger.opt(raw=True).info("-" * 80)
            logger.info("Calculating combined scores")
            (result_combined, weights_combined) = combined_score(
                self.runner_config.group_id, result_peptide, result_glycan
            )

            if isinstance(weights_combined, pd.DataFrame):
                logger.info(weights_combined)

            end_at_combined = time.time() - start_at_combined
            seconds = int(end_at_combined)
            msecs = int(1000 * (end_at_combined - seconds))
            logger.info(
                "combined scores calculated: %d seconds and %d msecs" % (seconds, msecs)
            )

            start_at_stats = time.time()
            logger.opt(raw=True).info("-" * 80)
            logger.info("Calculating error statistics")
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
            logger.info(
                "error statistics finished: %d seconds and %d msecs" % (seconds, msecs)
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
        if self.config.context == "score_learn":
            # We only want to save the weights in the context of learning, to avoid overwriting the weights in the context of applying weights
            self.writer.save_weights(weights)

        seconds = int(needed)
        msecs = int(1000 * (needed - seconds))

        logger.info("Total time: %d seconds and %d msecs wall time" % (seconds, msecs))

        if self.runner_config.classifier == "LDA":
            return self.config.extra_writes.get("trained_weights_path")
        elif self.runner_config.classifier == "SVM":
            return self.config.extra_writes.get("trained_weights_path")
        elif self.runner_config.classifier in ("XGBoost", "HistGradientBoosting"):
            return self.config.extra_writes.get(
                f"trained_model_path_{self.config.level}"
            )

    def print_summary(self, result):
        if result.summary_statistics is not None:
            logger.opt(raw=True).info("=" * 80)
            logger.opt(raw=True).info("\n")
            logger.opt(raw=True).info(result.summary_statistics)
            logger.opt(raw=True).info("\n")
            logger.opt(raw=True).info("=" * 80)
            logger.opt(raw=True).info("\n")


class PyProphetLearner(PyProphetRunner):
    """
    Implements the learning and scoring workflow for PyProphet.

    This class extends PyProphetRunner to include functionality for training a classifier
    and applying it to the input data.
    """

    def run_algo(self, part=None):
        """
        Runs the learning and scoring algorithm.

        Args:
            part (str, optional): Specifies the part of the workflow to run (e.g., "peptide", "glycan"). Specific for glycopeptide workflows.

        Returns:
            tuple: A tuple containing the result, scorer, and weights.
        """
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
    """
    Applies pre-trained weights to full/new datasets.

    This class extends PyProphetRunner to include functionality for loading pre-trained
    weights and applying them to input data.
    """

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
            if self.classifier in ("LDA", "SVM"):
                try:
                    self.persisted_weights = pd.read_csv(apply_weights, sep=",")
                    self.persisted_weights = self.persisted_weights[
                        self.persisted_weights["level"] == self.level
                    ]
                    if self.level != self.persisted_weights["level"].unique()[0]:
                        raise click.ClickException("Weights file has wrong level.")
                except Exception:
                    import traceback

                    traceback.print_exc()
                    raise
            elif self.classifier in ("XGBoost", "HistGradientBoosting"):
                with open(apply_weights, "rb") as file:
                    loaded_data = pickle.load(file)
                    # Check if this is new format (dict with metadata) or old format (just model)
                    if isinstance(loaded_data, dict) and "model" in loaded_data:
                        # New format with metadata
                        self.persisted_weights = loaded_data["model"]
                        stored_ss_main_score = loaded_data.get("ss_main_score", "auto")
                        stored_level = loaded_data.get("level", self.level)
                        stored_classifier = loaded_data.get("classifier", self.classifier)
                        
                        # Validate level matches
                        if stored_level != self.level:
                            raise click.ClickException(
                                f"Weights file was trained for level '{stored_level}' but you are applying to level '{self.level}'."
                            )
                        
                        # Validate classifier matches
                        if stored_classifier != self.classifier:
                            logger.warning(
                                f"Weights file was trained with classifier '{stored_classifier}' but you specified '{self.classifier}'. "
                                f"Using stored classifier '{stored_classifier}'."
                            )
                        
                        # Use stored ss_main_score if current config has default/auto
                        if config.runner.ss_main_score == "auto" and stored_ss_main_score != "auto":
                            logger.info(
                                f"Using stored ss_main_score='{stored_ss_main_score}' from weights file. "
                                f"Original training used this main score, applying it ensures feature alignment."
                            )
                            config.runner.ss_main_score = stored_ss_main_score
                        elif config.runner.ss_main_score != stored_ss_main_score:
                            logger.warning(
                                f"Current ss_main_score='{config.runner.ss_main_score}' differs from stored ss_main_score='{stored_ss_main_score}'. "
                                f"This may cause feature misalignment. Consider using --ss_main_score={stored_ss_main_score}"
                            )
                    else:
                        # Old format (backward compatibility) - just the model
                        self.persisted_weights = loaded_data
                        logger.warning(
                            "Loading weights from old format file (no metadata). "
                            "Feature alignment cannot be automatically verified. "
                            "Make sure to specify the same --ss_main_score as used during training."
                        )
        elif self.config.file_type == "osw":
            if self.classifier in ("LDA", "SVM"):
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
            elif self.classifier in ("XGBoost", "HistGradientBoosting"):
                try:
                    con = sqlite3.connect(apply_weights)

                    if not check_sqlite_table(con, "PYPROPHET_XGB"):
                        raise click.ClickException(
                            "PYPROPHET_XGB table is not present in file, cannot apply weights for XGBoost/HistGradientBoosting classifier! Make sure you have run the scoring on a subset of the data first, or that you supplied the right `--classifier` parameter."
                        )
                    data = con.execute(
                        "SELECT xgb FROM PYPROPHET_XGB WHERE LEVEL=='%s'" % self.level
                    ).fetchone()
                    con.close()
                    loaded_data = pickle.loads(data[0])
                    
                    # Check if this is new format (dict with metadata) or old format (just model)
                    if isinstance(loaded_data, dict) and "model" in loaded_data:
                        # New format with metadata
                        self.persisted_weights = loaded_data["model"]
                        stored_ss_main_score = loaded_data.get("ss_main_score", "auto")
                        stored_level = loaded_data.get("level", self.level)
                        stored_classifier = loaded_data.get("classifier", self.classifier)
                        
                        # Validate level matches
                        if stored_level != self.level:
                            raise click.ClickException(
                                f"Weights file was trained for level '{stored_level}' but you are applying to level '{self.level}'."
                            )
                        
                        # Validate classifier matches
                        if stored_classifier != self.classifier:
                            logger.warning(
                                f"Weights file was trained with classifier '{stored_classifier}' but you specified '{self.classifier}'. "
                                f"Using stored classifier '{stored_classifier}'."
                            )
                        
                        # Use stored ss_main_score if current config has default/auto
                        if config.runner.ss_main_score == "auto" and stored_ss_main_score != "auto":
                            logger.info(
                                f"Using stored ss_main_score='{stored_ss_main_score}' from weights file. "
                                f"Original training used this main score, applying it ensures feature alignment."
                            )
                            config.runner.ss_main_score = stored_ss_main_score
                        elif config.runner.ss_main_score != stored_ss_main_score:
                            logger.warning(
                                f"Current ss_main_score='{config.runner.ss_main_score}' differs from stored ss_main_score='{stored_ss_main_score}'. "
                                f"This may cause feature misalignment. Consider using --ss_main_score={stored_ss_main_score}"
                            )
                    else:
                        # Old format (backward compatibility) - just the model
                        self.persisted_weights = loaded_data
                        logger.warning(
                            "Loading weights from old format file (no metadata). "
                            "Feature alignment cannot be automatically verified. "
                            "Make sure to specify the same --ss_main_score as used during training."
                        )
                except Exception:
                    import traceback

                    traceback.print_exc()
                    raise

    def run_algo(self):
        """
        Runs the algorithm to apply pre-trained weights.

        Returns:
            tuple: A tuple containing the result, scorer, and weights.
        """
        (result, scorer, weights) = PyProphet(self.config).apply_weights(
            self.table, self.persisted_weights
        )
        return (result, scorer, weights)
