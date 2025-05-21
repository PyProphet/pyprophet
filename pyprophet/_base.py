from dataclasses import dataclass, field
import os
from .io.util import is_sqlite_file, is_parquet_file, is_valid_split_parquet_dir


@dataclass
class BaseIOConfig:
    """
    Base configuration class for I/O-related metadata.

    Attributes:
        infile (str): Path to the input file (e.g., .osw, .parquet).
        outfile (str): Path to the output file to be written.
        file_type (str): Type of the input file (e.g., 'osw', 'parquet', 'parquet_split', 'tsv').
        subsample_ratio (float): Subsampling ratio for large data.
        level (str): Scoring level (e.g., 'ms1', 'ms2', 'transition', 'alignment').
        context (str): Context or mode in which the reader/writer operates
                       (e.g., 'score', 'ipf', 'level_context').
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
        if is_sqlite_file(self.infile):
            self.file_type = "osw"
        elif is_parquet_file(self.infile):
            self.file_type = "parquet"
        elif is_valid_split_parquet_dir(self.infile):
            self.file_type = "parquet_split"
        else:
            self.file_type = "tsv"
        self.prefix = os.path.splitext(self.outfile)[0]
