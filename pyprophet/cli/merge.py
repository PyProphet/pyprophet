import os
import shutil
import time
import click
from loguru import logger

from .util import (
    CombinedGroup,
    AdvancedHelpCommand,
    write_logfile,
    measure_memory_usage_and_time,
)

from ..util import (
    merge_osw as _merge_osw,
)
from .._base import BaseIOConfig
from ..io._base import BaseSplitParquetWriter


def create_merge_group():
    @click.group(name="merge", cls=CombinedGroup)
    def merge():
        """
        Subcommands for merge files for different formats.
        """
        pass

    merge.add_command(merge_osw, name="osw")
    merge.add_command(merge_parquet, name="parquet")

    return merge


# Merging of multiple runs
@click.command(name="osw", cls=AdvancedHelpCommand)
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
@measure_memory_usage_and_time
def merge_osw(infiles, outfile, same_run, templatefile, merged_post_scored_runs):
    """
    Merge multiple OSW files and (for large experiments, it is recommended to subsample first).
    """

    if len(infiles) < 1:
        raise click.ClickException(
            "At least one PyProphet input file needs to be provided."
        )

    _merge_osw(infiles, outfile, templatefile, same_run, merged_post_scored_runs)


@click.command(name="parquet", cls=AdvancedHelpCommand)
@click.argument("infiles", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--out",
    "outfile",
    required=True,
    type=click.Path(exists=False),
    help="Merged parquet output file.",
)
@click.option(
    "--merge_transitions/--no-merge_transitions",
    default=False,
    is_flag=True,
    help="If the input is of type split_parquet / split_parquet_multi, merge the separate transition files into a single file as well.",
)
@measure_memory_usage_and_time
def merge_parquet(infiles, outfile, merge_transitions):
    """
    Merge multiple parquet files.
    """

    if len(infiles) < 1:
        raise click.ClickException(
            "At least one PyProphet input file needs to be provided."
        )

    if os.path.exists(outfile):
        logger.warning(
            f"{outfile} already exists, will overwrite/delete",
        )

        time.sleep(10)

        if os.path.isdir(outfile):
            shutil.rmtree(outfile)
        else:
            os.remove(outfile)

    config = BaseIOConfig(
        infile=infiles[0],
        outfile=outfile,
        subsample_ratio=1.0,  # Subsample ratio is not applicable for merging
        level="parquet",
        context="merge_parquet",
    )

    if config.file_type in ["parquet_split", "parquet_split_multi"]:
        writer = BaseSplitParquetWriter(config)
        writer.merge_files(merge_transitions)
    else:
        raise click.ClickException(
            "Expected input files to be of type 'parquet_split' or 'parquet_split_multi'. "
            f"Got {config.file_type} instead."
        )
