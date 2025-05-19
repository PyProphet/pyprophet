from dataclasses import dataclass, field
import os


@dataclass
class BaseIOConfig:
    """
    Base configuration class for I/O-related metadata.

    Attributes:
        infile (str): Path to the input file (e.g., .osw, .parquet).
        outfile (str): Path to the output file to be written.
        level (str): Scoring level (e.g., 'ms1', 'ms2', 'transition', 'alignment').
        context (str): Context or mode in which the reader/writer operates
                       (e.g., 'score', 'ipf', 'level_context').
        prefix (str): Automatically derived from outfile (e.g., 'results/output' for 'results/output.osw').
    """

    infile: str
    outfile: str
    level: str
    context: str
    prefix: str = field(init=False)

    def __post_init__(self):
        self.prefix = os.path.splitext(self.outfile)[0]
