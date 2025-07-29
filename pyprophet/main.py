import pathlib
import sqlite3

import click
import pandas as pd
from tabulate import tabulate

from .cli.export import create_export_group
from .cli.ipf import (
    glycoform as glycoform_command,
)
from .cli.ipf import (
    ipf as ipf_command,
)
from .cli.levels_context import (
    create_levels_context_group,
    peptide as peptide_command,
    protein as protein_command,
    gene as gene_command,
)
from .cli.merge import create_merge_group
from .cli.score import score as score_command
from .cli.util import (
    CombinedGroup,
    PythonLiteralOption,
    transform_subsample_ratio,
    transform_threads,
)
from .filter import filter_osw, filter_sqmass
from .io.util import check_sqlite_table
from .util import (
    backpropagate_oswr,
    reduce_osw,
    subsample_osw,
)
from .split import split_merged_parquet, split_osw

try:
    profile
except NameError:

    def profile(fun):
        return fun


# Changed chain to False to allow grouping related subcommands @singjc
@click.group(chain=False, cls=CombinedGroup)
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
@click.option(
    "--log-colorize/--no-log-colorize",
    default=True,
    help="Turn on/off colorized logging output.",
)
@click.pass_context
def cli(ctx, log_level, log_colorize):
    """
    PyProphet: Semi-supervised learning and scoring of OpenSWATH results.

    Visit http://openswath.org for usage instructions and help.
    """


# Semi-supervised learning and scoring of OpenSWATH results
cli.add_command(score_command, name="score")

# Peptidoform and Levels contexts for peptide/protein/gene infernce
levels_context = create_levels_context_group()
cli.add_command(levels_context)

# Exporters for OpenSWATH results
exporters = create_export_group()
cli.add_command(exporters, name="export")

# Merging of OpenSWATH files
mergers = create_merge_group()
cli.add_command(mergers, name="merge")


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
            raise click.ClickException(
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
        raise click.ClickException(
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
                f"Total peakgroups (q-value<{qt}): {len(peakgroups[peakgroups['QVALUE'] < qt]['FEATURE_ID'].drop_duplicates())}"
            )
            click.echo(f"Total peakgroups per run (q-value<{qt}):")
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
                f"Total peptides (global context) (q-value<{qt}): {len(peptides_global[peptides_global['QVALUE'] < qt]['PEPTIDE_ID'].drop_duplicates())}"
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
                f"Total proteins (global context) (q-value<{qt}): {len(proteins_global[proteins_global['QVALUE'] < qt]['PROTEIN_ID'].drop_duplicates())}"
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


# ----------------------------------------------------
# Deprecated commands to be removed in future versions

# We keep these commands for backward compatibility, but they are deprecated.
# To be removed in future versions of PyProphet. We first ensure documentation of use of the old commans is updated

cli.add_command(ipf_command, name="ipf")
ipf = cli.get_command(None, "ipf")
ipf.deprecated = True

# Peptide inference
cli.add_command(peptide_command, name="peptide")
peptide = cli.get_command(None, "peptide")
peptide.deprecated = True
# Protein inference
cli.add_command(protein_command, name="protein")
protein = cli.get_command(None, "protein")
protein.deprecated = True
# Gene inference
cli.add_command(gene_command, name="gene")
gene = cli.get_command(None, "gene")
gene.deprecated = True
