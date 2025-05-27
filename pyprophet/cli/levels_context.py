import click
from loguru import logger

from .util import (
    CombinedGroup,
    AdvancedHelpCommand,
    shared_statistics_options,
    write_logfile,
    measure_memory_usage_and_time,
)
from .._config import LevelContextIOConfig
from ..levels_contexts import (
    infer_glycopeptides,
    infer_peptides,
    infer_proteins,
    infer_genes,
)


def create_levels_context_group():
    @click.group(name="levels-context", cls=CombinedGroup)
    @click.pass_context
    def levels_context(ctx):
        """Subcommands for FDR estimation at different biological levels."""
        pass

    levels_context.add_command(peptide, name="peptide")
    levels_context.add_command(glycopeptide, name="glycopeptide")
    levels_context.add_command(gene, name="gene")
    levels_context.add_command(protein, name="protein")

    return levels_context


# Peptide-level inference
@click.command(cls=AdvancedHelpCommand)
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
@click.pass_context
@measure_memory_usage_and_time
@logger.catch(reraise=True)
def peptide(
    ctx,
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
    write_logfile(
        ctx.obj["LOG_LEVEL"],
        f"{config.prefix}_pyp_levels_context_peptide_{context}.log",
        ctx.obj["LOG_HEADER"],
    )
    infer_peptides(config)


# GlycoPeptide-level inference
@click.command(cls=AdvancedHelpCommand)
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
@click.pass_context
@measure_memory_usage_and_time
@logger.catch(reraise=True)
def glycopeptide(
    ctx,
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
    write_logfile(
        ctx.obj["LOG_LEVEL"],
        f"{config.prefix}_pyp_levels_context_glycopeptide_{context}.log",
        ctx.obj["LOG_HEADER"],
    )
    infer_glycopeptides(config)


# Gene-level inference
@click.command(cls=AdvancedHelpCommand)
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
@click.pass_context
@measure_memory_usage_and_time
@logger.catch(reraise=True)
def gene(
    ctx,
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
    write_logfile(
        ctx.obj["LOG_LEVEL"],
        f"{config.prefix}_pyp_levels_context_gene_{context}.log",
        ctx.obj["LOG_HEADER"],
    )
    infer_genes(config)


# Protein-level inference
@click.command(cls=AdvancedHelpCommand)
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
@click.pass_context
@measure_memory_usage_and_time
@logger.catch(reraise=True)
def protein(
    ctx,
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
    write_logfile(
        ctx.obj["LOG_LEVEL"],
        f"{config.prefix}_pyp_levels_context_protein_{context}.log",
        ctx.obj["LOG_HEADER"],
    )
    infer_proteins(config)
