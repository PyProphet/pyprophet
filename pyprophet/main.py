import sys
import os
import pathlib
import shutil
import ast
import time
import sqlite3
import pandas as pd
import numpy as np
import click
from loguru import logger
from functools import update_wrapper
from tabulate import tabulate

from hyperopt import hp

from ._config import RunnerIOConfig, IPFIOConfig, LevelContextIOConfig
from .io.util import check_sqlite_table, setup_logger
from .runner import PyProphetLearner, PyProphetWeightApplier
from .ipf import infer_peptidoforms
from .levels_contexts import (
    infer_glycopeptides,
    infer_peptides,
    infer_proteins,
    infer_genes,
    subsample_osw,
    reduce_osw,
    merge_osw,
    backpropagate_oswr,
)
from .glyco.glycoform import infer_glycoforms
from .split import split_merged_parquet, split_osw
from .export import export_tsv, export_score_plots
from .export_compound import export_compound_tsv
from .glyco.export import (
    export_tsv as export_glyco_tsv,
    export_score_plots as export_glyco_score_plots,
)
from .filter import filter_sqmass, filter_osw
from .data_handling import (
    transform_pi0_lambda,
    transform_threads,
    transform_subsample_ratio,
)
from .export_parquet import (
    export_to_parquet,
    convert_osw_to_parquet,
    convert_sqmass_to_parquet,
)


try:
    profile
except NameError:

    def profile(fun):
        return fun


class GlobalLogLevelGroup(click.Group):
    def invoke(self, ctx):
        log_level = ctx.params.get("log_level", "INFO").upper()
        setup_logger(log_level=log_level)
        ctx.obj = {"LOG_LEVEL": log_level}
        return super().invoke(ctx)


# Changed chain to False to allow grouping related subcommands @singjc
@click.group(chain=False, cls=GlobalLogLevelGroup)
@click.version_option()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        case_sensitive=False,
    ),
    help="Set global logging level.",
)
@click.pass_context
def cli(ctx, log_level):
    """
    PyProphet: Semi-supervised learning and scoring of OpenSWATH results.

    Visit http://openswath.org for usage instructions and help.
    """


# https://stackoverflow.com/a/47730333
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        if not isinstance(value, str):  # required for Click>=8.0.0
            return value
        try:
            return ast.literal_eval(value)
        except Exception:
            raise click.BadParameter(value)


def shared_statistics_options(func):
    """
    Decorator to add shared statistics options to a command.
    """
    options = [
        click.option(
            "--parametric/--no-parametric",
            default=False,
            show_default=True,
            help="Do parametric estimation of p-values.",
        ),
        click.option(
            "--pfdr/--no-pfdr",
            default=False,
            show_default=True,
            help="Compute positive false discovery rate (pFDR) instead of FDR.",
        ),
        click.option(
            "--pi0_lambda",
            default=[0.1, 0.5, 0.05],
            show_default=True,
            type=(float, float, float),
            help="Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.",
            callback=transform_pi0_lambda,
        ),
        click.option(
            "--pi0_method",
            default="bootstrap",
            show_default=True,
            type=click.Choice(["smoother", "bootstrap"]),
            help="Method for choosing tuning parameter in pi₀ estimation.",
        ),
        click.option(
            "--pi0_smooth_df",
            default=3,
            show_default=True,
            type=int,
            help="Degrees of freedom for smoother when estimating pi₀.",
        ),
        click.option(
            "--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0",
            default=False,
            show_default=True,
            help="Apply smoother to log(pi₀) values during estimation.",
        ),
        click.option(
            "--lfdr_truncate/--no-lfdr_truncate",
            show_default=True,
            default=True,
            help="If True, local FDR values > 1 are set to 1.",
        ),
        click.option(
            "--lfdr_monotone/--no-lfdr_monotone",
            show_default=True,
            default=True,
            help="Ensure local FDR values are non-decreasing with p-values.",
        ),
        click.option(
            "--lfdr_transformation",
            default="probit",
            show_default=True,
            type=click.Choice(["probit", "logit"]),
            help="Transformation applied to p-values for local FDR estimation.",
        ),
        click.option(
            "--lfdr_adj",
            default=1.5,
            show_default=True,
            type=float,
            help="Smoothing bandwidth multiplier used in density estimation.",
        ),
        click.option(
            "--lfdr_eps",
            default=np.power(10.0, -8),
            show_default=True,
            type=float,
            help="Threshold for p-value tails in local FDR calculation.",
        ),
    ]

    for option in reversed(options):
        func = option(func)
    return func


# PyProphet semi-supervised learning and scoring
@cli.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file. Valid formats are .osw, .parquet and .tsv.",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="PyProphet output file. Valid formats are .osw, .parquet and .tsv. Must be the same format as input file.",
)
@click.option(
    "--subsample_ratio",
    default=1.0,
    show_default=True,
    type=float,
    help="Subsampling ratio for large data. Use <1.0 to subsample precursors for semi-supervised learning, the learned weights will then be applied to the full data set.",
)
# Semi-supervised learning
@click.option(
    "--classifier",
    default="LDA",
    show_default=True,
    type=click.Choice(["LDA", "SVM", "XGBoost"]),
    help='Either a "LDA", "SVM" or "XGBoost" classifier is used for semi-supervised learning.',
)
@click.option(
    "--autotune/--no-autotune",
    default=False,
    show_default=True,
    help="Autotune hyperparameters for XGBoost/SVM.",
)
@click.option(
    "--apply_weights",
    type=click.Path(exists=True),
    help="Apply PyProphet score weights file instead of semi-supervised learning.",
)
@click.option(
    "--xeval_fraction",
    default=0.5,
    show_default=True,
    type=float,
    help="Data fraction used for cross-validation of semi-supervised learning step.",
)
@click.option(
    "--xeval_num_iter",
    default=10,
    show_default=True,
    type=int,
    help="Number of iterations for cross-validation of semi-supervised learning step.",
)
@click.option(
    "--ss_initial_fdr",
    default=0.15,
    show_default=True,
    type=float,
    help="Initial FDR cutoff for best scoring targets.",
)
@click.option(
    "--ss_iteration_fdr",
    default=0.05,
    show_default=True,
    type=float,
    help="Iteration FDR cutoff for best scoring targets.",
)
@click.option(
    "--ss_num_iter",
    default=10,
    show_default=True,
    type=int,
    help="Number of iterations for semi-supervised learning step.",
)
@click.option(
    "--ss_main_score",
    default="auto",
    show_default=True,
    type=str,
    help='Main score to start semi-supervised-learning. Default is set to auto, meaning each iteration of learning a dynamic main score selection process will occur. If you want to have a set starting main score for each learning iteration, you can set a specifc score, i.e. "var_xcorr_shape"',
)
@click.option(
    "--ss_score_filter",
    default="",
    help='Specify scores which should used for scoring. In addition specific predefined profiles can be used. For example for metabolomis data use "metabolomics".  Please specify any additional input as follows: "var_ms1_xcorr_coelution,var_library_corr,var_xcorr_coelution,etc."',
)
@click.option(
    "--ss_scale_features/--no-ss_scale_features",
    default=False,
    show_default=True,
    help="Scale features before semi-supervised learning.",
)
# Statistics
@click.option(
    "--group_id",
    default="group_id",
    show_default=True,
    type=str,
    help="Group identifier for calculation of statistics.",
)
@shared_statistics_options
# OpenSWATH options
@click.option(
    "--level",
    default="ms2",
    show_default=True,
    type=click.Choice(["ms1", "ms2", "ms1ms2", "transition", "alignment"]),
    help='Either "ms1", "ms2", "ms1ms2", "transition", or "alignment"; the data level selected for scoring. "ms1ms2 integrates both MS1- and MS2-level scores and can be used instead of "ms2"-level results."',
)
@click.option(
    "--add_alignment_features/--no-add_alignment_features",
    default=False,
    show_default=True,
    help="Add alignment features to scoring.",
)
# IPF options
@click.option(
    "--ipf_max_peakgroup_rank",
    default=1,
    show_default=True,
    type=int,
    help="Assess transitions only for candidate peak groups until maximum peak group rank.",
)
@click.option(
    "--ipf_max_peakgroup_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Assess transitions only for candidate peak groups until maximum posterior error probability.",
)
@click.option(
    "--ipf_max_transition_isotope_overlap",
    default=0.5,
    show_default=True,
    type=float,
    help="Maximum isotope overlap to consider transitions in IPF.",
)
@click.option(
    "--ipf_min_transition_sn",
    default=0,
    show_default=True,
    type=float,
    help="Minimum log signal-to-noise level to consider transitions in IPF. Set -1 to disable this filter.",
)
# Glyco/GproDIA Options
@click.option(
    "--glyco/--no-glyco",
    default=False,
    show_default=True,
    help="Whether glycopeptide scoring should be enabled.",
)
@click.option(
    "--density_estimator",
    default="gmm",
    show_default=True,
    type=click.Choice(["kde", "gmm"]),
    help='Either kernel density estimation ("kde") or Gaussian mixture model ("gmm") is used for score density estimation.',
)
@click.option(
    "--grid_size",
    default=256,
    show_default=True,
    type=int,
    help="Number of d-score cutoffs to build grid coordinates for local FDR calculation.",
)
# TRIC
@click.option(
    "--tric_chromprob/--no-tric_chromprob",
    default=False,
    show_default=True,
    help="Whether chromatogram probabilities for TRIC should be computed.",
)
# Visualization
@click.option(
    "--color_palette",
    default="normal",
    show_default=True,
    type=click.Choice(["normal", "protan", "deutran", "tritan"]),
    help="Color palette to use in reports.",
)
@click.option(
    "--main_score_selection_report/--no-main_score_selection_report",
    default=False,
    show_default=True,
    help="Generate a report for main score selection process.",
)
# Processing
@click.option(
    "--threads",
    default=1,
    show_default=True,
    type=int,
    help="Number of threads used for semi-supervised learning. -1 means all available CPUs.",
    callback=transform_threads,
)
@click.option(
    "--test/--no-test",
    default=False,
    show_default=True,
    help="Run in test mode with fixed seed.",
)
@click.pass_context
@logger.catch(reraise=True)
def score(
    ctx,
    infile,
    outfile,
    subsample_ratio,
    classifier,
    autotune,
    apply_weights,
    xeval_fraction,
    xeval_num_iter,
    ss_initial_fdr,
    ss_iteration_fdr,
    ss_num_iter,
    ss_main_score,
    ss_score_filter,
    ss_scale_features,
    group_id,
    parametric,
    pfdr,
    pi0_lambda,
    pi0_method,
    pi0_smooth_df,
    pi0_smooth_log_pi0,
    lfdr_truncate,
    lfdr_monotone,
    lfdr_transformation,
    lfdr_adj,
    lfdr_eps,
    level,
    add_alignment_features,
    ipf_max_peakgroup_rank,
    ipf_max_peakgroup_pep,
    ipf_max_transition_isotope_overlap,
    ipf_min_transition_sn,
    glyco,
    density_estimator,
    grid_size,
    tric_chromprob,
    color_palette,
    main_score_selection_report,
    threads,
    test,
):
    """
    Conduct semi-supervised learning and error-rate estimation for MS1, MS2 and transition-level data.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    config = RunnerIOConfig.from_cli_args(
        infile,
        outfile,
        subsample_ratio,
        level,
        "score_learn",
        classifier,
        autotune,
        xeval_fraction,
        xeval_num_iter,
        ss_initial_fdr,
        ss_iteration_fdr,
        ss_num_iter,
        ss_main_score,
        ss_score_filter,
        ss_scale_features,
        group_id,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        lfdr_truncate,
        lfdr_monotone,
        lfdr_transformation,
        lfdr_adj,
        lfdr_eps,
        ipf_max_peakgroup_rank,
        ipf_max_peakgroup_pep,
        ipf_max_transition_isotope_overlap,
        ipf_min_transition_sn,
        add_alignment_features,
        glyco,
        density_estimator,
        grid_size,
        tric_chromprob,
        threads,
        test,
        color_palette,
        main_score_selection_report,
    )

    # Validate file type and subsample ratio, subsample_ratio is currently only applicateble for "parquet_split", "parquet_split_multi". If this combination is not met, throw warning and set subsample_ratio to 1.0
    if (
        config.file_type not in ["parquet", "parquet_split", "parquet_split_multi"]
        and subsample_ratio < 1.0
    ):
        logger.warning(
            "Semi-supervised learning on a subset of the data, and then applying the weights to the full data is currently only supported for `parquet_split` and `parquet_split_multi` files.\nFor `osw`, you need to manually subsample the osw using the `subsample` module.\nSetting subsample_ratio to 1.0.",
        )
        config.subsample_ratio = 1.0

    if not apply_weights:
        if config.subsample_ratio < 1.0:
            logger.info(
                f"Conducting {level} semi-supervised learning on {config.subsample_ratio * 100}% of the data.",
            )
            weights_path = PyProphetLearner(config).run()
            # Apply weights from subsampled result to full infile

            logger.info(
                f"Info: Applying {level} weights from {weights_path} to the full data set.",
            )
            config.subsample_ratio = 1.0
            config.context = "score_apply"
            PyProphetWeightApplier(weights_path, config).run()
        else:
            logger.info(
                f"Conducting {level} semi-supervised learning.",
            )
            PyProphetLearner(config).run()
    else:
        logger.info(
            f"Applying {level} weights from {apply_weights} to the full data set.",
        )
        PyProphetWeightApplier(apply_weights, config).run()


# IPF
@cli.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file. Valid formats are .osw, .parquet (produced by export_parquet with `--scoring_format`)",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="PyProphet output file. Valid formats are .osw, .parquet. Must be the same format as input file.",
)
# IPF parameters
@click.option(
    "--ipf_ms1_scoring/--no-ipf_ms1_scoring",
    default=True,
    show_default=True,
    help="Use MS1 precursor data for IPF.",
)
@click.option(
    "--ipf_ms2_scoring/--no-ipf_ms2_scoring",
    default=True,
    show_default=True,
    help="Use MS2 precursor data for IPF.",
)
@click.option(
    "--ipf_h0/--no-ipf_h0",
    default=True,
    show_default=True,
    help="Include possibility that peak groups are not covered by peptidoform space.",
)
@click.option(
    "--ipf_grouped_fdr/--no-ipf_grouped_fdr",
    default=False,
    show_default=True,
    help="[Experimental] Compute grouped FDR instead of pooled FDR to better support data where peak groups are evaluated to originate from very heterogeneous numbers of peptidoforms.",
)
@click.option(
    "--ipf_max_precursor_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored precursors in IPF.",
)
@click.option(
    "--ipf_max_peakgroup_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored peak groups in IPF.",
)
@click.option(
    "--ipf_max_precursor_peakgroup_pep",
    default=0.4,
    show_default=True,
    type=float,
    help="Maximum BHM layer 1 integrated precursor peakgroup PEP to consider in IPF.",
)
@click.option(
    "--ipf_max_transition_pep",
    default=0.6,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored transitions in IPF.",
)
@click.option(
    "--propagate_signal_across_runs/--no-propagate_signal_across_runs",
    default=False,
    show_default=True,
    help="Propagate signal across runs (requires running alignment).",
)
@click.option(
    "--ipf_max_alignment_pep",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for good alignments.",
)
@click.option(
    "--across_run_confidence_threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for propagating signal across runs for aligned features.",
)
def ipf(
    infile,
    outfile,
    ipf_ms1_scoring,
    ipf_ms2_scoring,
    ipf_h0,
    ipf_grouped_fdr,
    ipf_max_precursor_pep,
    ipf_max_peakgroup_pep,
    ipf_max_precursor_peakgroup_pep,
    ipf_max_transition_pep,
    propagate_signal_across_runs,
    ipf_max_alignment_pep,
    across_run_confidence_threshold,
):
    """
    Infer peptidoforms after scoring of MS1, MS2 and transition-level data.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    config = IPFIOConfig.from_cli_args(
        infile,
        outfile,
        1,  # Subsample ratio is not applicable for IPF
        "ipf",  # Level is not applicable for IPF
        "ipf",
        ipf_ms1_scoring,
        ipf_ms2_scoring,
        ipf_h0,
        ipf_grouped_fdr,
        ipf_max_precursor_pep,
        ipf_max_peakgroup_pep,
        ipf_max_precursor_peakgroup_pep,
        ipf_max_transition_pep,
        propagate_signal_across_runs,
        ipf_max_alignment_pep,
        across_run_confidence_threshold,
    )

    infer_peptidoforms(config)


# Infer glycoforms
@cli.command()
@click.option(
    "--in", "infile", required=True, type=click.Path(exists=True), help="Input file."
)
@click.option("--out", "outfile", type=click.Path(exists=False), help="Output file.")
@click.option(
    "--ms1_precursor_scoring/--no-ms1_precursor_scoring",
    default=True,
    show_default=True,
    help="Use MS1 precursor data for glycoform inference.",
)
@click.option(
    "--ms2_precursor_scoring/--no-ms2_precursor_scoring",
    default=True,
    show_default=True,
    help="Use MS2 precursor data for glycoform inference.",
)
@click.option(
    "--grouped_fdr/--no-grouped_fdr",
    default=False,
    show_default=True,
    help="[Experimental] Compute grouped FDR instead of pooled FDR to better support data where peak groups are evaluated to originate from very heterogeneous numbers of glycoforms.",
)
@click.option(
    "--max_precursor_pep",
    default=1,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored precursors.",
)
@click.option(
    "--max_peakgroup_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored peak groups.",
)
@click.option(
    "--max_precursor_peakgroup_pep",
    default=1,
    show_default=True,
    type=float,
    help="Maximum BHM layer 1 integrated precursor peakgroup PEP to consider.",
)
@click.option(
    "--max_transition_pep",
    default=0.6,
    show_default=True,
    type=float,
    help="Maximum PEP to consider scored transitions.",
)
@click.option(
    "--use_glycan_composition/--use_glycan_struct",
    "use_glycan_composition",
    default=True,
    show_default=True,
    help="Compute glycoform-level FDR based on glycan composition or struct.",
)
@click.option(
    "--ms1_mz_window",
    default=10,
    show_default=True,
    type=float,
    help="MS1 m/z window in Thomson or ppm.",
)
@click.option(
    "--ms1_mz_window_unit",
    default="ppm",
    show_default=True,
    type=click.Choice(["ppm", "Da", "Th"]),
    help="MS1 m/z window unit.",
)
@click.option(
    "--propagate_signal_across_runs/--no-propagate_signal_across_runs",
    default=False,
    show_default=True,
    help="Propagate signal across runs (requires running alignment).",
)
@click.option(
    "--max_alignment_pep",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for good alignments.",
)
@click.option(
    "--across_run_confidence_threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Maximum PEP to consider for propagating signal across runs for aligned features.",
)
def glycoform(
    infile,
    outfile,
    ms1_precursor_scoring,
    ms2_precursor_scoring,
    grouped_fdr,
    max_precursor_pep,
    max_peakgroup_pep,
    max_precursor_peakgroup_pep,
    max_transition_pep,
    use_glycan_composition,
    ms1_mz_window,
    ms1_mz_window_unit,
    propagate_signal_across_runs,
    max_alignment_pep,
    across_run_confidence_threshold,
):
    """
    Infer glycoforms after scoring of MS1, MS2 and transition-level data.
    """

    if outfile is None:
        outfile = infile

    infer_glycoforms(
        infile=infile,
        outfile=outfile,
        ms1_precursor_scoring=ms1_precursor_scoring,
        ms2_precursor_scoring=ms2_precursor_scoring,
        grouped_fdr=grouped_fdr,
        max_precursor_pep=max_precursor_pep,
        max_peakgroup_pep=max_peakgroup_pep,
        max_precursor_peakgroup_pep=max_precursor_peakgroup_pep,
        max_transition_pep=max_transition_pep,
        use_glycan_composition=use_glycan_composition,
        ms1_mz_window=ms1_mz_window,
        ms1_mz_window_unit=ms1_mz_window_unit,
        propagate_signal_across_runs=propagate_signal_across_runs,
        max_alignment_pep=max_alignment_pep,
        across_run_confidence_threshold=across_run_confidence_threshold,
    )


@cli.group(name="levels-context")
def levels_context():
    """
    Subcommands for FDR estimation at different biological levels.
    """
    pass


# Peptide-level inference
@levels_context.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file. Valid formats are .osw, .parquet (produced by export_parquet with `--scoring_format`)",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="PyProphet output file.  Valid formats are .osw, .parquet. Must be the same format as input file.",
)
# Context
@click.option(
    "--context",
    default="run-specific",
    show_default=True,
    type=click.Choice(["run-specific", "experiment-wide", "global"]),
    help="Context to estimate peptide-level FDR control.",
)
# Statistics
@shared_statistics_options
# Visualization
@click.option(
    "--color_palette",
    default="normal",
    show_default=True,
    type=click.Choice(["normal", "protan", "deutran", "tritan"]),
    help="Color palette to use in reports.",
)
def peptide(
    infile,
    outfile,
    context,
    parametric,
    pfdr,
    pi0_lambda,
    pi0_method,
    pi0_smooth_df,
    pi0_smooth_log_pi0,
    lfdr_truncate,
    lfdr_monotone,
    lfdr_transformation,
    lfdr_adj,
    lfdr_eps,
    color_palette,
):
    """
    Infer peptides and conduct error-rate estimation in different contexts.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    config = LevelContextIOConfig.from_cli_args(
        infile,
        outfile,
        1,  # Subsample ratio is not applicable for peptide-level inference
        "peptide",
        "levels_context",
        context,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        lfdr_truncate,
        lfdr_monotone,
        lfdr_transformation,
        lfdr_adj,
        lfdr_eps,
        color_palette,
        None,
        None,
    )

    infer_peptides(config)


# GlycoPeptide-level inference
@levels_context.command()
@click.option(
    "--in", "infile", required=True, type=click.Path(exists=True), help="Input file."
)
@click.option("--out", "outfile", type=click.Path(exists=False), help="Output file.")
@click.option(
    "--context",
    default="run-specific",
    show_default=True,
    type=click.Choice(["run-specific", "experiment-wide", "global"]),
    help="Context to estimate glycopeptide-level FDR control.",
)
@click.option(
    "--density_estimator",
    default="gmm",
    show_default=True,
    type=click.Choice(["kde", "gmm"]),
    help='Either kernel density estimation ("kde") or Gaussian mixture model ("gmm") is used for score density estimation.',
)
@click.option(
    "--grid_size",
    default=256,
    show_default=True,
    type=int,
    help="Number of d-score cutoffs to build grid coordinates for local FDR calculation.",
)
@shared_statistics_options
def glycopeptide(
    infile,
    outfile,
    context,
    density_estimator,
    grid_size,
    parametric,
    pfdr,
    pi0_lambda,
    pi0_method,
    pi0_smooth_df,
    pi0_smooth_log_pi0,
    lfdr_truncate,
    lfdr_monotone,
    **kwargs,  # unused kwargs for shared_statistics_options
):
    """
    Infer glycopeptides and conduct error-rate estimation in different contexts.
    """
    if outfile is None:
        outfile = infile

    config = LevelContextIOConfig.from_cli_args(
        infile,
        outfile,
        1,  # Subsample ratio is not applicable for glycopeptide-level inference
        "glycopeptide",
        "levels_context",
        context,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        lfdr_truncate,
        lfdr_monotone,
        "probit",
        1.5,
        1e-8,
        "normal",
        density_estimator,
        grid_size,
    )

    infer_glycopeptides(config)


# Gene-level inference
@levels_context.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file.  Valid formats are .osw, .parquet (produced by export_parquet with `--scoring_format`)",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="PyProphet output file.  Valid formats are .osw, .parquet. Must be the same format as input file.",
)
# Context
@click.option(
    "--context",
    default="run-specific",
    show_default=True,
    type=click.Choice(["run-specific", "experiment-wide", "global"]),
    help="Context to estimate gene-level FDR control.",
)
# Statistics
@shared_statistics_options
# Visualization
@click.option(
    "--color_palette",
    default="normal",
    show_default=True,
    type=click.Choice(["normal", "protan", "deutran", "tritan"]),
    help="Color palette to use in reports.",
)
def gene(
    infile,
    outfile,
    context,
    parametric,
    pfdr,
    pi0_lambda,
    pi0_method,
    pi0_smooth_df,
    pi0_smooth_log_pi0,
    lfdr_truncate,
    lfdr_monotone,
    lfdr_transformation,
    lfdr_adj,
    lfdr_eps,
    color_palette,
):
    """
    Infer genes and conduct error-rate estimation in different contexts.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    config = LevelContextIOConfig.from_cli_args(
        infile,
        outfile,
        1,  # Subsample ratio is not applicable for gene-level inference
        "gene",
        "levels_context",
        context,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        lfdr_truncate,
        lfdr_monotone,
        lfdr_transformation,
        lfdr_adj,
        lfdr_eps,
        color_palette,
        None,
        None,
    )

    infer_genes(config)


# Protein-level inference
@levels_context.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file.  Valid formats are .osw, .parquet (produced by export_parquet with `--scoring_format`)",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="PyProphet output file.  Valid formats are .osw, .parquet. Must be the same format as input file.",
)
# Context
@click.option(
    "--context",
    default="run-specific",
    show_default=True,
    type=click.Choice(["run-specific", "experiment-wide", "global"]),
    help="Context to estimate protein-level FDR control.",
)
# Statistics
@shared_statistics_options
# Visualization
@click.option(
    "--color_palette",
    default="normal",
    show_default=True,
    type=click.Choice(["normal", "protan", "deutran", "tritan"]),
    help="Color palette to use in reports.",
)
def protein(
    infile,
    outfile,
    context,
    parametric,
    pfdr,
    pi0_lambda,
    pi0_method,
    pi0_smooth_df,
    pi0_smooth_log_pi0,
    lfdr_truncate,
    lfdr_monotone,
    lfdr_transformation,
    lfdr_adj,
    lfdr_eps,
    color_palette,
):
    """
    Infer proteins and conduct error-rate estimation in different contexts.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    config = LevelContextIOConfig.from_cli_args(
        infile,
        outfile,
        1,  # Subsample ratio is not applicable for gene-level inference
        "protein",
        "levels_context",
        context,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        lfdr_truncate,
        lfdr_monotone,
        lfdr_transformation,
        lfdr_adj,
        lfdr_eps,
        color_palette,
        None,
        None,
    )

    infer_proteins(config)


# Subsample OpenSWATH file to minimum for integrated scoring
@cli.command()
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="OpenSWATH input file.",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="Subsampled OSWS output file.",
)
@click.option(
    "--subsample_ratio",
    default=1,
    show_default=True,
    type=float,
    help="Subsample ratio used per input file.",
    callback=transform_subsample_ratio,
)
@click.option(
    "--test/--no-test",
    default=False,
    show_default=True,
    help="Run in test mode with fixed seed.",
)
def subsample(infile, outfile, subsample_ratio, test):
    """
    Subsample OpenSWATH file to minimum for integrated scoring
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    subsample_osw(infile, outfile, subsample_ratio, test)


# Reduce scored PyProphet file to minimum for global scoring
@cli.command()
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="Scored PyProphet input file.",
)
@click.option(
    "--out", "outfile", type=click.Path(exists=False), help="Reduced OSWR output file."
)
def reduce(infile, outfile):
    """
    Reduce scored PyProphet file to minimum for global scoring
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    reduce_osw(infile, outfile)


# Merging of multiple runs
@cli.command()
@click.argument("infiles", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--out",
    "outfile",
    required=True,
    type=click.Path(exists=False),
    help="Merged OSW output file.",
)
@click.option(
    "--same_run/--no-same_run",
    default=False,
    help="Assume input files are from same run (deletes run information).",
)
@click.option(
    "--template",
    "templatefile",
    required=True,
    type=click.Path(exists=False),
    help="Template OSW file.",
)
@click.option(
    "--merged_post_scored_runs",
    is_flag=True,
    help="Merge OSW output files that have already been scored.",
)
def merge(infiles, outfile, same_run, templatefile, merged_post_scored_runs):
    """
    Merge multiple OSW files and (for large experiments, it is recommended to subsample first).
    """

    if len(infiles) < 1:
        raise click.ClickException(
            "At least one PyProphet input file needs to be provided."
        )

    merge_osw(infiles, outfile, templatefile, same_run, merged_post_scored_runs)


# Spliting of a merge osw into single runs
@cli.command()
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="Merged OSW input file.",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="Single run OSW output file.",
)
@click.option(
    "--threads",
    default=-1,
    show_default=True,
    type=int,
    help="Number of threads used for splitting. -1 means all available CPUs.",
    callback=transform_threads,
)
def split(infile, outfile, threads):
    """
    Split a merged OSW file into single runs.
    """
    if infile.endswith(".osw"):
        split_osw(infile, threads)
    else:
        split_merged_parquet(infile, outfile)


# Backpropagate multi-run peptide and protein scores to single files
@cli.command()
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="Single run PyProphet input file.",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="Single run (with multi-run scores) PyProphet output file.",
)
@click.option(
    "--apply_scores",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet multi-run scores file to apply.",
)
def backpropagate(infile, outfile, apply_scores):
    """
    Backpropagate multi-run peptide and protein scores to single files
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    backpropagate_oswr(infile, outfile, apply_scores)


@cli.group(name="export")
def export():
    """
    Subcommands for exporting between different formats.
    """
    pass


# Export TSV
@export.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file.",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="Output TSV/CSV (matrix, legacy_split, legacy_merged) file.",
)
@click.option(
    "--format",
    default="legacy_split",
    show_default=True,
    type=click.Choice(["matrix", "legacy_split", "legacy_merged"]),
    help="Export format, either matrix, legacy_split/legacy_merged (mProphet/PyProphet) or score_plots format.",
)
@click.option(
    "--csv/--no-csv",
    "outcsv",
    default=False,
    show_default=True,
    help="Export CSV instead of TSV file.",
)
# Context
@click.option(
    "--transition_quantification/--no-transition_quantification",
    default=True,
    show_default=True,
    help="[format: legacy] Report aggregated transition-level quantification.",
)
@click.option(
    "--max_transition_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="[format: legacy] Maximum PEP to retain scored transitions for quantification (requires transition-level scoring).",
)
@click.option(
    "--ipf",
    default="peptidoform",
    show_default=True,
    type=click.Choice(["peptidoform", "augmented", "disable"]),
    help='[format: matrix/legacy] Should IPF results be reported if present? "peptidoform": Report results on peptidoform-level, "augmented": Augment OpenSWATH results with IPF scores, "disable": Ignore IPF results',
)
@click.option(
    "--ipf_max_peptidoform_pep",
    default=0.4,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] IPF: Filter results to maximum run-specific peptidoform-level PEP.",
)
@click.option(
    "--max_rs_peakgroup_qvalue",
    default=0.05,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum run-specific peak group-level q-value.",
)
@click.option(
    "--peptide/--no-peptide",
    default=True,
    show_default=True,
    help="Append peptide-level error-rate estimates if available.",
)
@click.option(
    "--max_global_peptide_qvalue",
    default=0.01,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum global peptide-level q-value.",
)
@click.option(
    "--protein/--no-protein",
    default=True,
    show_default=True,
    help="Append protein-level error-rate estimates if available.",
)
@click.option(
    "--max_global_protein_qvalue",
    default=0.01,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum global protein-level q-value.",
)
def tsv(
    infile,
    outfile,
    format,
    outcsv,
    transition_quantification,
    max_transition_pep,
    ipf,
    ipf_max_peptidoform_pep,
    max_rs_peakgroup_qvalue,
    peptide,
    max_global_peptide_qvalue,
    protein,
    max_global_protein_qvalue,
):
    """
    Export Proteomics/Peptidoform TSV/CSV tables
    """
    if outfile is None:
        if outcsv:
            outfile = infile.split(".osw")[0] + ".csv"
        else:
            outfile = infile.split(".osw")[0] + ".tsv"
    else:
        outfile = outfile

    export_tsv(
        infile,
        outfile,
        format,
        outcsv,
        transition_quantification,
        max_transition_pep,
        ipf,
        ipf_max_peptidoform_pep,
        max_rs_peakgroup_qvalue,
        peptide,
        max_global_peptide_qvalue,
        protein,
        max_global_protein_qvalue,
    )


# Export to Parquet
@export.command()
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet OSW or sqMass input file.",
)
@click.option(
    "--out",
    "outfile",
    required=False,
    type=click.Path(exists=False),
    help="Output parquet file.",
)
@click.option(
    "--oswfile",
    "oswfile",
    required=False,
    type=click.Path(exists=False),
    help="PyProphet OSW file. Only required when converting sqMass to parquet.",
)
@click.option(
    "--transitionLevel",
    "transitionLevel",
    is_flag=True,
    help="Whether to export transition level data as well",
)
@click.option(
    "--onlyFeatures",
    "onlyFeatures",
    is_flag=True,
    help="Only include precursors that have a corresponding feature",
)
@click.option(
    "--noDecoys",
    "noDecoys",
    is_flag=True,
    help="Do not include decoys in the exported data",
)
# Convert to scoring format
@click.option(
    "--scoring_format",
    "scoring_format",
    is_flag=True,
    help="Convert OSW to parquet format that is compatible with the scoring/inference modules",
)
@click.option(
    "--split_transition_data/--no-split_transition_data",
    "split_transition_data",
    default=False,
    show_default=True,
    help="Split transition data into a separate parquet (default: True).",
)
@click.option(
    "--split_runs/--no-split_runs",
    "split_runs",
    default=False,
    show_default=True,
    help="Split runs into separate parquet files/directories (default: False).",
)
@click.option(
    "--compression",
    "compression",
    default="zstd",
    show_default=True,
    type=click.Choice(
        ["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]
    ),
    help="Compression algorithm to use for parquet file.",
)
@click.option(
    "--compression_level",
    "compression_level",
    default=11,
    show_default=True,
    type=int,
    help="Compression level to use for parquet file.",
)
def parquet(
    infile,
    outfile,
    oswfile,
    transitionLevel,
    onlyFeatures,
    noDecoys,
    scoring_format,
    split_transition_data,
    split_runs,
    compression,
    compression_level,
):
    """
    Export OSW or sqMass to parquet format
    """
    # Check if the input file has an .osw extension
    if infile.endswith(".osw"):
        if scoring_format:
            click.echo("Info: Will export OSW to parquet scoring format")
            if os.path.exists(outfile):
                click.echo(
                    click.style(
                        f"Warn: {outfile} already exists, will overwrite/delete",
                        fg="yellow",
                    )
                )

                time.sleep(10)

                if os.path.isdir(outfile):
                    shutil.rmtree(outfile)
                else:
                    os.remove(outfile)

            if split_transition_data:
                click.echo(
                    f"Info: {outfile} will be a directory containing a separate precursors_features.parquet and a transition_features.parquet"
                )

            start = time.time()
            convert_osw_to_parquet(
                infile,
                outfile,
                compression_method=compression,
                compression_level=compression_level,
                split_transition_data=split_transition_data,
                split_runs=split_runs,
            )
            end = time.time()
            click.echo(f"Info: {outfile} written in {end-start:.4f} seconds.")

        else:
            if transitionLevel:
                click.echo("Info: Will export transition level data")
            if outfile is None:
                outfile = infile.split(".osw")[0] + ".parquet"
            if os.path.exists(outfile):
                overwrite = click.confirm(
                    f"{outfile} already exists, would you like to overwrite?"
                )
                if not overwrite:
                    raise click.ClickException(f"Aborting: {outfile} already exists!")
            click.echo("Info: Parquet file will be written to {}".format(outfile))
            export_to_parquet(
                os.path.abspath(infile),
                os.path.abspath(outfile),
                transitionLevel,
                onlyFeatures,
                noDecoys,
            )
    elif infile.endswith(".sqmass") or infile.endswith(".sqMass"):
        click.echo("Info: Will export sqMass to parquet")
        if os.path.exists(outfile):
            click.echo(
                click.style(
                    f"Warn: {outfile} already exists, will overwrite", fg="yellow"
                )
            )
        start = time.time()
        convert_sqmass_to_parquet(
            infile,
            outfile,
            oswfile,
            compression_method=compression,
            compression_level=compression_level,
        )
        end = time.time()
        click.echo(f"Info: {outfile} written in {end-start:.4f} seconds.")
    else:
        raise click.ClickException("Input file must be of type .osw or .sqmass/.sqMass")


# Export Compound TSV
@export.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file.",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="Output TSV/CSV (matrix, legacy_merged) file.",
)
@click.option(
    "--format",
    default="legacy_merged",
    show_default=True,
    type=click.Choice(["matrix", "legacy_merged"]),
    help="Export format, either matrix, legacy_merged (PyProphet) or score_plots format.",
)
@click.option(
    "--csv/--no-csv",
    "outcsv",
    default=False,
    show_default=True,
    help="Export CSV instead of TSV file.",
)
# Context
@click.option(
    "--max_rs_peakgroup_qvalue",
    default=0.05,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum run-specific peak group-level q-value.",
)
def compound(infile, outfile, format, outcsv, max_rs_peakgroup_qvalue):
    """
    Export Compound TSV/CSV tables
    """
    if outfile is None:
        if outcsv:
            outfile = infile.split(".osw")[0] + ".csv"
        else:
            outfile = infile.split(".osw")[0] + ".tsv"
    else:
        outfile = outfile

    export_compound_tsv(infile, outfile, format, outcsv, max_rs_peakgroup_qvalue)


# Export Glycoform TSV
@export.command()
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file.",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="Output TSV/CSV (matrix, legacy_split, legacy_merged) file.",
)
@click.option(
    "--format",
    default="legacy_split",
    show_default=True,
    type=click.Choice(["matrix", "legacy_split", "legacy_merged"]),
    help="Export format, either matrix, legacy_split/legacy_merged (mProphet/PyProphet) format.",
)
@click.option(
    "--csv/--no-csv",
    "outcsv",
    default=False,
    show_default=True,
    help="Export CSV instead of TSV file.",
)
# Context
@click.option(
    "--transition_quantification/--no-transition_quantification",
    default=True,
    show_default=True,
    help="[format: legacy] Report aggregated transition-level quantification.",
)
@click.option(
    "--max_transition_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="[format: legacy] Maximum PEP to retain scored transitions for quantification (requires transition-level scoring).",
)
@click.option(
    "--max_rs_peakgroup_qvalue",
    default=0.05,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum run-specific peak group-level q-value.",
)
@click.option(
    "--glycoform_match_precursor",
    default="glycan_composition",
    show_default=True,
    type=click.Choice(["exact", "glycan_composition", "none"]),
    help="[format: matrix/legacy] Export glycoform results with glycan matched with precursor-level results.",
)
@click.option(
    "--max_glycoform_pep",
    default=1,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum glycoform PEP.",
)
@click.option(
    "--max_glycoform_qvalue",
    default=0.05,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum glycoform q-value.",
)
@click.option(
    "--glycopeptide/--no-glycopeptide",
    default=True,
    show_default=True,
    help="Append glycopeptide-level error-rate estimates if available.",
)
@click.option(
    "--max_global_glycopeptide_qvalue",
    default=0.01,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum global glycopeptide-level q-value.",
)
def glyco(
    infile,
    outfile,
    format,
    outcsv,
    transition_quantification,
    max_transition_pep,
    max_rs_peakgroup_qvalue,
    glycoform_match_precursor,
    max_glycoform_pep,
    max_glycoform_qvalue,
    glycopeptide,
    max_global_glycopeptide_qvalue,
):
    """
    Export Gylcoform TSV/CSV tables
    """

    if outfile is None:
        if outcsv:
            outfile = infile.split(".osw")[0] + ".csv"
        else:
            outfile = infile.split(".osw")[0] + ".tsv"
    else:
        outfile = outfile

    export_glyco_tsv(
        infile,
        outfile,
        format=format,
        outcsv=outcsv,
        transition_quantification=transition_quantification,
        max_transition_pep=max_transition_pep,
        glycoform=glycoform,
        glycoform_match_precursor=glycoform_match_precursor,
        max_glycoform_pep=max_glycoform_pep,
        max_glycoform_qvalue=max_glycoform_qvalue,
        max_rs_peakgroup_qvalue=max_rs_peakgroup_qvalue,
        glycopeptide=glycopeptide,
        max_global_glycopeptide_qvalue=max_global_glycopeptide_qvalue,
    )


# Export score plots
@export.command()
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet OSW input file.",
)
# Glycoform
@click.option(
    "--glycoform/--no-glycoform",
    "glycoform",
    default=False,
    show_default=True,
    help="Export glycoform score plots.",
)
def score_plots(infile):
    """
    Export score plots
    """
    if infile.endswith(".osw"):
        if not glycoform:
            export_score_plots(infile)
        else:
            export_glyco_score_plots(infile)
    else:
        raise click.ClickException("Input file must be of type .osw")


# Filter sqMass or OSW files
@cli.command()
# SqMass Filter File handling
@click.argument("sqldbfiles", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--in",
    "infile",
    required=False,
    default=None,
    show_default=True,
    type=click.Path(exists=True),
    help="PyProphet input file.",
)
@click.option(
    "--max_precursor_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to retain scored precursors in sqMass.",
)
@click.option(
    "--max_peakgroup_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to retain scored peak groups in sqMass.",
)
@click.option(
    "--max_transition_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Maximum PEP to retain scored transitions in sqMass.",
)
# OSW Filter File Handling
@click.option(
    "--remove_decoys/--no-remove_decoys",
    "remove_decoys",
    default=True,
    show_default=True,
    help="Remove Decoys from OSW file.",
)
@click.option(
    "--omit_tables",
    default="[]",
    show_default=True,
    cls=PythonLiteralOption,
    help="""Tables in the database you do not want to copy over to filtered file. i.e. `--omit_tables '["FEATURE_TRANSITION", "SCORE_TRANSITION"]'`""",
)
@click.option(
    "--max_gene_fdr",
    default=None,
    show_default=True,
    type=float,
    help="Maximum QVALUE to retain scored genes in OSW.  [default: None]",
)
@click.option(
    "--max_protein_fdr",
    default=None,
    show_default=True,
    type=float,
    help="Maximum QVALUE to retain scored proteins in OSW.  [default: None]",
)
@click.option(
    "--max_peptide_fdr",
    default=None,
    show_default=True,
    type=float,
    help="Maximum QVALUE to retain scored peptides in OSW.  [default: None]",
)
@click.option(
    "--max_ms2_fdr",
    default=None,
    show_default=True,
    type=float,
    help="Maximum QVALUE to retain scored MS2 Features in OSW.  [default: None]",
)
@click.option(
    "--keep_naked_peptides",
    default="[]",
    show_default=True,
    cls=PythonLiteralOption,
    help="""Filter for specific UNMODIFIED_PEPTIDES. i.e. `--keep_naked_peptides '["ANSSPTTNIDHLK", "ESTAEPDSLSR"]'`""",
)
@click.option(
    "--run_ids",
    default="[]",
    show_default=True,
    cls=PythonLiteralOption,
    help="""Filter for specific RUN_IDs. i.e. `--run_ids '["8889961272137748833", "8627438106464817423"]'`""",
)
def filter(
    sqldbfiles,
    infile,
    max_precursor_pep,
    max_peakgroup_pep,
    max_transition_pep,
    remove_decoys,
    omit_tables,
    max_gene_fdr,
    max_protein_fdr,
    max_peptide_fdr,
    max_ms2_fdr,
    keep_naked_peptides,
    run_ids,
):
    """
    Filter sqMass files or osw files
    """

    if all(
        [pathlib.PurePosixPath(file).suffix.lower() == ".sqmass" for file in sqldbfiles]
    ):
        if infile is None and len(keep_naked_peptides) == 0:
            click.ClickException(
                "If you are filtering sqMass files, you need to provide a PyProphet file via `--in` flag or you need to provide a list of naked peptide sequences to filter for."
            )
        filter_sqmass(
            sqldbfiles,
            infile,
            max_precursor_pep,
            max_peakgroup_pep,
            max_transition_pep,
            keep_naked_peptides,
            remove_decoys,
        )
    elif all(
        [pathlib.PurePosixPath(file).suffix.lower() == ".osw" for file in sqldbfiles]
    ):
        filter_osw(
            sqldbfiles,
            remove_decoys,
            omit_tables,
            max_gene_fdr,
            max_protein_fdr,
            max_peptide_fdr,
            max_ms2_fdr,
            keep_naked_peptides,
            run_ids,
        )
    else:
        click.ClickException(
            f"There seems to be something wrong with the input sqlite db files. Make sure they are all either sqMass files or all OSW files, these are mutually exclusive.\nYour input files: {sqldbfiles}"
        )


# Print statistics
@cli.command()
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file.",
)
def statistics(infile):
    """
    Print PyProphet statistics
    """

    con = sqlite3.connect(infile)

    qts = [0.01, 0.05, 0.10]

    for qt in qts:
        if check_sqlite_table(con, "SCORE_MS2"):
            peakgroups = pd.read_sql(
                "SELECT * FROM SCORE_MS2 INNER JOIN FEATURE ON SCORE_MS2.feature_id = FEATURE.id INNER JOIN PRECURSOR ON FEATURE.precursor_id = PRECURSOR.id INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID WHERE RANK==1 AND DECOY==0;",
                con,
            )

            click.echo(
                "Total peakgroups (q-value<%s): %s"
                % (
                    qt,
                    len(
                        peakgroups[peakgroups["QVALUE"] < qt][
                            ["FEATURE_ID"]
                        ].drop_duplicates()
                    ),
                )
            )
            click.echo("Total peakgroups per run (q-value<%s):" % qt)
            click.echo(
                tabulate(
                    peakgroups[peakgroups["QVALUE"] < qt]
                    .groupby(["FILENAME"])["FEATURE_ID"]
                    .nunique()
                    .reset_index(),
                    showindex=False,
                )
            )
            click.echo(10 * "=")

        if check_sqlite_table(con, "SCORE_PEPTIDE"):
            peptides_global = pd.read_sql(
                'SELECT * FROM SCORE_PEPTIDE INNER JOIN PEPTIDE ON SCORE_PEPTIDE.peptide_id = PEPTIDE.id WHERE CONTEXT=="global" AND DECOY==0;',
                con,
            )
            peptides = pd.read_sql(
                "SELECT * FROM SCORE_PEPTIDE INNER JOIN PEPTIDE ON SCORE_PEPTIDE.peptide_id = PEPTIDE.id INNER JOIN RUN ON SCORE_PEPTIDE.RUN_ID = RUN.ID WHERE DECOY==0;",
                con,
            )

            click.echo(
                "Total peptides (global context) (q-value<%s): %s"
                % (
                    qt,
                    len(
                        peptides_global[peptides_global["QVALUE"] < qt][
                            ["PEPTIDE_ID"]
                        ].drop_duplicates()
                    ),
                )
            )
            click.echo(
                tabulate(
                    peptides[peptides["QVALUE"] < qt]
                    .groupby(["FILENAME"])["PEPTIDE_ID"]
                    .nunique()
                    .reset_index(),
                    showindex=False,
                )
            )
            click.echo(10 * "=")

        if check_sqlite_table(con, "SCORE_PROTEIN"):
            proteins_global = pd.read_sql(
                'SELECT * FROM SCORE_PROTEIN INNER JOIN PROTEIN ON SCORE_PROTEIN.protein_id = PROTEIN.id WHERE CONTEXT=="global" AND DECOY==0;',
                con,
            )
            proteins = pd.read_sql(
                "SELECT * FROM SCORE_PROTEIN INNER JOIN PROTEIN ON SCORE_PROTEIN.protein_id = PROTEIN.id INNER JOIN RUN ON SCORE_PROTEIN.RUN_ID = RUN.ID WHERE DECOY==0;",
                con,
            )

            click.echo(
                "Total proteins (global context) (q-value<%s): %s"
                % (
                    qt,
                    len(
                        proteins_global[proteins_global["QVALUE"] < qt][
                            ["PROTEIN_ID"]
                        ].drop_duplicates()
                    ),
                )
            )
            click.echo(
                tabulate(
                    proteins[proteins["QVALUE"] < qt]
                    .groupby(["FILENAME"])["PROTEIN_ID"]
                    .nunique()
                    .reset_index(),
                    showindex=False,
                )
            )
            click.echo(10 * "=")
