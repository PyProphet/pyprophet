"""
This module implements context-specific inference workflows for genes, proteins, peptides,
and glycopeptides in targeted proteomics and glycoproteomics data analysis.

It provides functions for statistical error estimation, FDR control, and reporting
at different levels of biological organization. The workflows support global,
experiment-wide, and run-specific contexts.

Functions:
    - statistics_report: Generates error statistics and updates the input data with FDR-related metrics.
    - infer_genes: Performs gene-level inference based on the specified context.
    - infer_proteins: Performs protein-level inference based on the specified context.
    - infer_peptides: Performs peptide-level inference based on the specified context.
    - infer_glycopeptides: Performs glycopeptide-level inference based on the specified context.
"""

import click
from loguru import logger

from ._config import LevelContextIOConfig
from .glyco.stats import statistics_report as glyco_statistics_report
from .io.dispatcher import ReaderDispatcher, WriterDispatcher
from .stats import (
    error_statistics,
    final_err_table,
    lookup_values_from_error_table,
    summary_err_table,
)


def statistics_report(
    data,
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
    writer,
):
    """
    Generates error statistics and updates the input data with FDR-related metrics.

    Args:
        data (pd.DataFrame): Input data containing scores and decoy labels.
        parametric (bool): Whether to use parametric FDR estimation.
        pfdr (bool): Whether to use pFDR estimation.
        pi0_lambda (list): Lambda values for pi0 estimation.
        pi0_method (str): Method for pi0 estimation.
        pi0_smooth_df (int): Degrees of freedom for pi0 smoothing.
        pi0_smooth_log_pi0 (bool): Whether to log-transform pi0 values.
        lfdr_truncate (bool): Whether to truncate local FDR values.
        lfdr_monotone (bool): Whether to enforce monotonicity in local FDR values.
        lfdr_transformation (str): Transformation method for local FDR.
        lfdr_adj (float): Adjustment factor for local FDR.
        lfdr_eps (float): Epsilon value for local FDR.
        writer (Writer): Writer object for saving reports.

    Returns:
        pd.DataFrame: Updated data with FDR-related metrics.
    """
    error_stat, pi0 = error_statistics(
        data[data.decoy == 0]["score"],
        data[data.decoy == 1]["score"],
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        True,
        lfdr_truncate,
        lfdr_monotone,
        lfdr_transformation,
        lfdr_adj,
        lfdr_eps,
    )

    stat_table = final_err_table(error_stat)
    summary_table = summary_err_table(error_stat)

    # print summary table
    logger.opt(raw=True).info("=" * 80)
    logger.opt(raw=True).info("\n")
    logger.opt(raw=True).info(summary_table)
    logger.opt(raw=True).info("\n")
    logger.opt(raw=True).info("=" * 80)
    logger.opt(raw=True).info("\n")

    p_values, s_values, peps, q_values = lookup_values_from_error_table(
        data["score"].values, error_stat
    )
    data["p_value"] = p_values
    data["s_value"] = s_values
    data["q_value"] = q_values
    data["pep"] = peps

    writer._write_levels_context_pdf_report(data.copy(), stat_table, pi0)

    return data


def infer_genes(
    config: LevelContextIOConfig,
):
    """
    Performs gene-level inference based on the specified context.

    Args:
        config (LevelContextIOConfig): Configuration object for the workflow.
    """
    context = config.context_fdr
    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    if context in ["global", "experiment-wide", "run-specific"]:
        data = reader.read()
    else:
        raise click.ClickException("Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]

    if context == "run-specific":
        data = data.groupby("run_id").apply(
            statistics_report,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )
    elif context in ["global", "experiment-wide"]:
        data = statistics_report(
            data,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )

    # Store results
    writer.save_results(data)


def infer_proteins(
    config: LevelContextIOConfig,
):
    """
    Performs protein-level inference based on the specified context.

    Args:
        config (LevelContextIOConfig): Configuration object for the workflow.
    """

    context = config.context_fdr
    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    if context in ["global", "experiment-wide", "run-specific"]:
        data = reader.read()
    else:
        raise click.ClickException("Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]

    if context == "run-specific":
        data = data.groupby("run_id").apply(
            statistics_report,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )
    elif context in ["global", "experiment-wide"]:
        data = statistics_report(
            data,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )

    # Store results
    writer.save_results(data)


def infer_peptides(config: LevelContextIOConfig):
    """
    Performs peptide-level inference based on the specified context.

    Args:
        config (LevelContextIOConfig): Configuration object for the workflow.
    """

    context = config.context_fdr
    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    if context in ["global", "experiment-wide", "run-specific"]:
        data = reader.read()
    else:
        raise click.ClickException("Unspecified context selected.")

    if context == "run-specific":
        data = data.groupby("run_id").apply(
            statistics_report,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )

    elif context in ["global", "experiment-wide"]:
        data = statistics_report(
            data,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )

    # store data in table
    writer.save_results(data)


def infer_glycopeptides(
    config: LevelContextIOConfig,
):
    """
    Performs glycopeptide-level inference based on the specified context.

    Adapted from: https://github.com/lmsac/GproDIA/blob/main/src/glycoprophet/level_contexts.py

    Args:
        config (LevelContextIOConfig): Configuration object for the workflow.
    """
    context = config.context_fdr
    reader = ReaderDispatcher.get_reader(config)

    if context in ["global", "experiment-wide", "run-specific"]:
        data = reader.read()
    else:
        raise click.ClickException("Unspecified context selected.")

    if context == "run-specific":
        data = (
            data.groupby("run_id")
            .apply(
                glyco_statistics_report,
                config.outfile,
                context,
                "glycopeptide",
                density_estimator=config.density_estimator,
                grid_size=config.grid_size,
                parametric=config.error_estimation_config.parametric,
                pfdr=config.error_estimation_config.pfdr,
                pi0_lambda=config.error_estimation_config.pi0_lambda,
                pi0_method=config.error_estimation_config.pi0_method,
                pi0_smooth_df=config.error_estimation_config.pi0_smooth_df,
                pi0_smooth_log_pi0=config.error_estimation_config.pi0_smooth_log_pi0,
                lfdr_truncate=config.error_estimation_config.lfdr_truncate,
                lfdr_monotone=config.error_estimation_config.lfdr_monotone,
                # lfdr_transformation=lfdr_transformation,
                # lfdr_adj=lfdr_adj, lfdr_eps=lfdr_eps
            )
            .reset_index()
        )

    elif context in ["global", "experiment-wide"]:
        data = glyco_statistics_report(
            data,
            config.outfile,
            context,
            "glycopeptide",
            density_estimator=config.density_estimator,
            grid_size=config.grid_size,
            parametric=config.parametric,
            pfdr=config.pfdr,
            pi0_lambda=config.pi0_lambda,
            pi0_method=config.pi0_method,
            pi0_smooth_df=config.pi0_smooth_df,
            pi0_smooth_log_pi0=config.pi0_smooth_log_pi0,
            lfdr_truncate=config.lfdr_truncate,
            lfdr_monotone=config.lfdr_monotone,
            # lfdr_transformation=lfdr_transformation,
            # lfdr_adj=lfdr_adj, lfdr_eps=lfdr_eps
        )

    # store data in table
    writer = WriterDispatcher.get_writer(config)
    writer.save_results(data)
