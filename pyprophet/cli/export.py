import os
import shutil
import time
from pathlib import Path

import click
from loguru import logger

from .._config import ExportIOConfig
from ..export.export_compound import export_compound_tsv
from ..export.export_report import (
    export_score_plots as _export_score_plots,
)
from ..export.export_report import (
    export_scored_report as _export_scored_report,
)
from ..glyco.export import (
    export_score_plots as export_glyco_score_plots,
)
from ..glyco.export import (
    export_tsv as export_glyco_tsv,
)
from ..io.dispatcher import ReaderDispatcher, WriterDispatcher
from .util import (
    AdvancedHelpCommand,
    CombinedGroup,
    measure_memory_usage_and_time,
)


def create_export_group():
    @click.group(name="export", cls=CombinedGroup)
    def export():
        """
        Subcommands for exporting between different formats.
        """
        pass

    export.add_command(export_tsv, name="tsv")
    export.add_command(export_library, name='library')
    export.add_command(export_matrix, name="matrix")
    export.add_command(export_parquet, name="parquet")
    export.add_command(export_compound, name="compound")
    export.add_command(export_glyco, name="glyco")
    export.add_command(export_score_plots, name="score-plots")
    export.add_command(export_scored_report, name="score-report")

    return export


# Export TSV
@click.command(name="tsv", cls=AdvancedHelpCommand)
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
    help="Output TSV/CSV (legacy_split, legacy_merged) file.",
)
@click.option(
    "--format",
    default="legacy_merged",
    show_default=True,
    type=click.Choice(["legacy_split", "legacy_merged"]),
    help="Export format, either legacy_split/legacy_merged (mProphet/PyProphet).",
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
@measure_memory_usage_and_time
def export_tsv(
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

    config = ExportIOConfig(
        infile=infile,
        outfile=outfile,
        subsample_ratio=1.0,  # Not used in export
        level="export",
        context="export",
        export_format=format,
        out_type="csv" if outcsv else "tsv",
        transition_quantification=transition_quantification,
        max_transition_pep=max_transition_pep,
        ipf=ipf,
        ipf_max_peptidoform_pep=ipf_max_peptidoform_pep,
        max_rs_peakgroup_qvalue=max_rs_peakgroup_qvalue,
        peptide=peptide,
        max_global_peptide_qvalue=max_global_peptide_qvalue,
        protein=protein,
        max_global_protein_qvalue=max_global_protein_qvalue,
    )

    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    df = reader.read()
    writer.export_results(df)


# Export Quantification Matrix TSV
@click.command(name="matrix", cls=AdvancedHelpCommand)
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
    help="Output TSV/CSV file.",
)
@click.option(
    "--level",
    default="peptide",
    show_default=True,
    type=click.Choice(["precursor", "peptide", "protein", "gene"]),
    help="Export quantification level, either precursor, peptide, protein, or gene.",
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
    "--max_global_peptide_qvalue",
    default=0.01,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum global peptide-level q-value.",
)
@click.option(
    "--max_global_protein_qvalue",
    default=0.01,
    show_default=True,
    type=float,
    help="[format: matrix/legacy] Filter results to maximum global protein-level q-value.",
)
@click.option(
    "--top_n",
    default=3,
    show_default=True,
    type=int,
    help="[format: matrix/legacy] Number of top intense features to use for summarization",
)
@click.option(
    "--consistent_top/--no-consistent_top",
    "consistent_top",
    default=True,
    show_default=True,
    help="[format: matrix/legacy] Whether to use same top features across all runs",
)
@click.option(
    "--normalization",
    default="none",
    show_default=True,
    type=click.Choice(["none", "median", "medianmedian", "quantile"]),
    help="[format: matrix/legacy] Normalization method to apply to the quantification matrix.",
)
@measure_memory_usage_and_time
def export_matrix(
    infile,
    outfile,
    level,
    outcsv,
    transition_quantification,
    max_transition_pep,
    ipf,
    ipf_max_peptidoform_pep,
    max_rs_peakgroup_qvalue,
    max_global_peptide_qvalue,
    max_global_protein_qvalue,
    top_n,
    consistent_top,
    normalization,
):
    """
    Export Proteomics/Peptidoform Quantification Matrix
    """
    if outfile is None:
        if outcsv:
            outfile = infile.split(".osw")[0] + ".csv"
        else:
            outfile = infile.split(".osw")[0] + ".tsv"
    else:
        outfile = outfile

    config = ExportIOConfig(
        infile=infile,
        outfile=outfile,
        subsample_ratio=1.0,  # Not used in export
        level=level,
        context="export",
        export_format="matrix",
        out_type="csv" if outcsv else "tsv",
        transition_quantification=transition_quantification,
        max_transition_pep=max_transition_pep,
        ipf=ipf,
        ipf_max_peptidoform_pep=ipf_max_peptidoform_pep,
        max_rs_peakgroup_qvalue=max_rs_peakgroup_qvalue,
        peptide=True,
        max_global_peptide_qvalue=max_global_peptide_qvalue,
        protein=True,
        max_global_protein_qvalue=max_global_protein_qvalue,
        top_n=top_n,
        consistent_top=consistent_top,
        normalization=normalization,
    )

    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    df = reader.read()
    writer.export_quant_matrix(df)

# Export to Library to be used in OpenSWATH
@click.command(name="library", cls=AdvancedHelpCommand)
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet OSW input file.",
)
@click.option(
    "--out",
    "outfile",
    required=True, # need to name the library or else get error in os.path.splittext line 75, in __post_init__in _base.
    type=click.Path(exists=False),
    help="Output tsv library.",
)
@click.option(
    "--max_peakgroup_qvalue",
    default=0.01,
    show_default=True,
    type=float,
    help="Filter results to maximum run-specific peak group-level q-value, using values greater than final statistical filtering (in most cases > 0.01), may lead to an overestimation in identification rates. If there are multiple runs with the same precursors, the run with the lowest q value is used",
)
@click.option(
    "--max_global_peptide_qvalue",
    default=0.01,
    show_default=True,
    type=float,
    help="Filter results to maximum global peptide-level q-value, using values greater than final statistical filtering (in most cases > 0.01), may lead to an overestimation in identification rates."
)
@click.option(
    "--max_global_protein_qvalue",
    default=0.01,
    show_default=True,
    type=float,
    help="Filter results to maximum global protein-level q-value, using values greater than final statistical filtering (in most cases > 0.01), may lead to an overestimation in identification rates."
)
@click.option(
    "--rt_calibration/--no-rt_calibration",
    default=True,
    show_default=True,
    help="Use empirical RT values as oppose to the original library RT values."
)
@click.option(
    "--im_calibration/--no-im_calibration",
    default=True,
    show_default=True,
    help="Use empirical IM values as oppose to the original library IM values."
)
@click.option(
    "--intensity_calibration/--no-intensity_calibration",
    default=True,
    show_default=True,
    help="Use empirical intensity values as oppose to the original library intensity values."
)
@click.option(
    "--min_fragments",
    default=4,
    show_default=True,
    type=int,
    help="Minimum number of fragments required to include the peak group in the library, only relevant if intensityCalibration is True."
)
@click.option(
    "--keep_decoys/--no-keep_decoys",
    default=False,
    show_default=True,
    type=bool,
    help="(Experimental) Whether to keep decoys in the exported library. Default is False, which means decoys are filtered out. Only keeps decoys passing thresholds specified above"
)
@click.option(
    "--rt_unit",
    default="iRT",
    show_default=True,
    type=click.Choice(["iRT", "RT"]),
    help='Unit of retention time in the library, only relevant if rt_calibration is True. If "iRT" is selected, the retention times will be scaled to the iRT scale (0-100) in the library.',
    hidden=True
)
@click.option(
    "--test/--no-test",
    default=False,
    show_default=True,
    help="Enable test mode with deterministic behavior, test mode will sort libraries by precursor, fragmentType, fragmentSeriesNumber and fragmentCharge")
@measure_memory_usage_and_time
def export_library(
    infile,
    outfile,
    max_peakgroup_qvalue,
    max_global_peptide_qvalue,
    max_global_protein_qvalue,
    rt_calibration,
    im_calibration,
    intensity_calibration,
    min_fragments,
    keep_decoys,
    rt_unit,
    test
):
    """
    Export OSW to tsv library format
    """
    config = ExportIOConfig(
        infile=infile,
        outfile=outfile,
        subsample_ratio=1.0,  # Not used in export
        level="export",
        context="export",
        export_format="library",
        out_type="tsv",
        max_rs_peakgroup_qvalue=max_peakgroup_qvalue,
        max_global_peptide_qvalue=max_global_peptide_qvalue,
        max_global_protein_qvalue=max_global_protein_qvalue,
        rt_calibration=rt_calibration,
        im_calibration=im_calibration,
        intensity_calibration=intensity_calibration,
        min_fragments=min_fragments,
        keep_decoys=keep_decoys,
        rt_unit=rt_unit,
        test=test
    )

    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    df = reader.read()
    writer.clean_and_export_library(df)

# Export to Parquet
@click.command(name="parquet", cls=AdvancedHelpCommand)
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
    "--pqpfile",
    "pqpfile",
    required=False,
    type=click.Path(exists=False),
    help="PyProphet PQP file. Only required when converting sqMass to parquet.",
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
@measure_memory_usage_and_time
def export_parquet(
    infile,
    outfile,
    pqpfile,
    transitionLevel,
    onlyFeatures,
    noDecoys,
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
        logger.info("Will export OSW to parquet scoring format")
        if os.path.exists(outfile):
            logger.warning(
                f"Warn: {outfile} already exists, will overwrite/delete in 10 seconds",
            )

            time.sleep(10)

            if os.path.isdir(outfile):
                shutil.rmtree(outfile)
            else:
                os.remove(outfile)

        if split_transition_data:
            logger.info(
                f"{outfile} will be a directory containing a separate precursors_features.parquet and a transition_features.parquet"
            )

        config = ExportIOConfig(
            infile=infile,
            outfile=outfile,
            subsample_ratio=1.0,  # Not used in export
            level="osw",
            context="export",
            export_format="parquet" if not split_transition_data else "parquet_split",
            split_transition_data=split_transition_data,
            split_runs=split_runs,
            compression_method=compression,
            compression_level=compression_level,
        )

        writer = WriterDispatcher.get_writer(config)
        writer.export()

    elif infile.endswith(".sqmass") or infile.endswith(".sqMass"):
        logger.info("Will export sqMass to parquet")
        if os.path.exists(outfile):
            logger.info(
                f"Warn: {outfile} already exists, will overwrite/delete in 10 seconds",
            )

            time.sleep(10)

            if os.path.isdir(outfile):
                shutil.rmtree(outfile)
            else:
                os.remove(outfile)

        config = ExportIOConfig(
            infile=infile,
            outfile=outfile,
            subsample_ratio=1.0,  # Not used in export
            level="sqmass",
            context="export",
            export_format="parquet",
            pqp_file=pqpfile,
            compression_method=compression,
            compression_level=compression_level,
        )

        writer = WriterDispatcher.get_writer(config)
        writer.export()
    else:
        raise click.ClickException("Input file must be of type .osw or .sqmass/.sqMass")


# Export Compound TSV
@click.command(name="compound", cls=AdvancedHelpCommand)
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
@measure_memory_usage_and_time
def export_compound(infile, outfile, format, outcsv, max_rs_peakgroup_qvalue):
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
@click.command(name="glyco", cls=AdvancedHelpCommand)
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
@measure_memory_usage_and_time
def export_glyco(
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
        glycoform=True,
        glycoform_match_precursor=glycoform_match_precursor,
        max_glycoform_pep=max_glycoform_pep,
        max_glycoform_qvalue=max_glycoform_qvalue,
        max_rs_peakgroup_qvalue=max_rs_peakgroup_qvalue,
        glycopeptide=glycopeptide,
        max_global_glycopeptide_qvalue=max_global_glycopeptide_qvalue,
    )


# Export score plots
@click.command(name="score-plots", cls=AdvancedHelpCommand)
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
@measure_memory_usage_and_time
def export_score_plots(infile, glycoform):
    """
    Export score plots
    """
    if infile.endswith(".osw"):
        if not glycoform:
            _export_score_plots(infile)
        else:
            export_glyco_score_plots(infile)
    else:
        raise click.ClickException("Input file must be of type .osw")


# Export score plots
@click.command(name="score-report", cls=AdvancedHelpCommand)
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet OSW input file.",
)
@measure_memory_usage_and_time
def export_scored_report(infile):
    """
    Export report with scored results from a PyProphet input file.
    """
    # Get file prefix (path/basename without extension), extension may be .osw, .parquet, or .tsv.
    outfile = Path(infile).stem + "_score_plots.pdf"
    logger.info(f"Exporting score plots to {outfile}")
    _export_scored_report(infile, outfile)
