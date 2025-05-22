from typing import Literal
import pandas as pd
import click

from .._base import BaseReader, BaseWriter, BaseIOConfig
from ...report import save_report


class TSVReader(BaseReader):
    """
    Class for reading and processing data from OpenSWATH results stored in a tsv format.

    The TSVReader class provides methods to read different levels of data from the split parquet files and process it accordingly.
    It supports reading data for semi-supervised learning, IPF analysis, context level analysis.

    Attributes:
        infile (str): Input file path.
        outfile (str): Output file path.
        classifier (str): Classifier used for semi-supervised learning.
        level (str): Level used in semi-supervised learning (e.g., 'ms1', 'ms2', 'ms1ms2', 'transition', 'alignment'), or context level used peptide/protein/gene inference (e.g., 'global', 'experiment-wide', 'run-specific').
        glyco (bool): Flag indicating whether analysis is glycoform-specific.

    Methods:
        read(): Read data from the input file based on the alogorithm.
    """

    def __init__(self, config: BaseIOConfig):
        super().__init__(config)

    def read(
        self, level: Literal["peakgroup_precursor", "transition", "alignment"]
    ) -> pd.DataFrame:
        raise NotImplementedError("IPF read method is not implemented for TSVReader")


class TSVWriter(BaseWriter):
    """
    Class for writing OpenSWATH results to a tsv format.

    Attributes:
        infile (str): Input file path.
        outfile (str): Output file path.
        classifier (str): Classifier used for semi-supervised learning.
        level (str): Level used in semi-supervised learning (e.g., 'ms1', 'ms2', 'ms1ms2', 'transition', 'alignment'), or context level used peptide/protein/gene inference (e.g., 'global', 'experiment-wide', 'run-specific').
        glyco (bool): Flag indicating whether analysis is glycoform-specific.

    Methods:
        save_results(result, pi0): Save the results to the output file based on the module using this class.
        save_weights(weights): Save the weights to the output file.
    """

    def __init__(self, config: BaseIOConfig):
        super().__init__(config)

    def save_results(self, result):
        raise NotImplementedError(
            "IPF save_results method is not implemented for TSVWriter"
        )
