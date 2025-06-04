from dataclasses import dataclass, field
import os
import copy
import sys

from loguru import logger
from .io.util import (
    is_tsv_file,
    is_sqlite_file,
    is_parquet_file,
    is_valid_single_split_parquet_dir,
    is_valid_multi_split_parquet_dir,
)


@dataclass
class BaseIOConfig:
    """
    Base configuration class for I/O-related metadata and common attributes used across algorithms.

    Attributes:
        infile (str): Path to the input file (e.g., .osw, .parquet).
        outfile (str): Path to the output file to be written.
        file_type (str): Type of the input file (e.g., 'osw', 'parquet', 'parquet_split', 'tsv').
        subsample_ratio (float): Subsampling ratio for large data.
        level (str): Scoring level (e.g., 'ms1', 'ms2', 'transition', 'alignment').
        context (str): Context or mode in which the reader/writer operates
                       (e.g., 'score_learn', 'score_apply', 'ipf', 'level_context').
        prefix (str): Automatically derived from outfile (e.g., 'results/output' for 'results/output.osw').
    """

    infile: str
    outfile: str
    file_type: str = field(init=False)
    subsample_ratio: float
    level: str
    context: str
    prefix: str = field(init=False)

    def __post_init__(self):
        """
        Initialize the file_type and prefix attributes based on the input file.
        """

        infile = self.infile

        if is_sqlite_file(infile):
            if infile.endswith(".osw"):
                self.file_type = "osw"
            elif infile.endswith(".sqmass") or infile.endswith(".sqMass"):
                self.file_type = "sqmass"
        elif is_parquet_file(infile):
            self.file_type = "parquet"
        elif is_valid_single_split_parquet_dir(infile):
            self.file_type = "parquet_split"
        elif is_valid_multi_split_parquet_dir(infile):
            self.file_type = "parquet_split_multi"
        elif is_tsv_file(infile):
            self.file_type = "tsv"
        else:
            logger.critical(
                f"Failed to infer file type for: {infile}. Supported formats are: "
                ".osw, .sqmass/.sqMass, .parquet, split parquet directories (.oswpq/oswpqd), or .tsv files.\n"
                f"  - is_sqlite_file: {is_sqlite_file(infile)}\n"
                f"  - endswith .osw: {infile.endswith('.osw')}\n"
                f"  - endswith .sqmass/.sqMass: {infile.endswith('.sqmass') or infile.endswith('.sqMass')}\n"
                f"  - is_parquet_file: {is_parquet_file(infile)}\n"
                f"  - is_valid_single_split_parquet_dir: {is_valid_single_split_parquet_dir(infile)}\n"
                f"  - is_valid_multi_split_parquet_dir: {is_valid_multi_split_parquet_dir(infile)}\n"
                f"  - is_tsv_file: {is_tsv_file(infile)}"
            )

            sys.exit(1)

        self.prefix = os.path.splitext(
            self.outfile
        )[
            0
        ]  # TODO: use pathlib instead to avoid potential cases where the outfile is a directory (split_parquet) file.oswpq/, which would result the prefix in being file.oswpq instead of file

    def __str__(self):
        return (
            f"BaseIOConfig(\ninfile='{self.infile}'\noutfile='{self.outfile}'\n"
            f"file_type='{self.file_type}'\nsubsample_ratio={self.subsample_ratio}\n"
            f"level='{self.level}'\ncontext='{self.context}'\nprefix='{self.prefix}')"
        )

    def __repr__(self):
        return (
            f"BaseIOConfig(infile='{self.infile}', outfile='{self.outfile}', "
            f"subsample_ratio={self.subsample_ratio}, level='{self.level}', "
            f"context='{self.context}')"
        )

    def copy(self):
        """
        Return a deep copy of the config object.
        """
        return copy.deepcopy(self)
