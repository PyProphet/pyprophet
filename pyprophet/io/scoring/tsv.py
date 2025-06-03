import pandas as pd
import click
from loguru import logger

from .._base import BaseReader, BaseWriter
from ..._config import RunnerIOConfig
from ...report import save_report


class TSVReader(BaseReader):
    """
    Class for reading and processing data from OpenSWATH results stored in a tsv format.

    The TSVReader class provides methods to read different levels of data from tsv feature format.
    It supports reading data for semi-supervised learning only.

    Attributes:
        infile (str): Input file path.
        outfile (str): Output file path.
        classifier (str): Classifier used for semi-supervised learning.
        level (str): Level used in semi-supervised learning (e.g., 'ms1', 'ms2', 'ms1ms2', 'transition', 'alignment'), or context level used peptide/protein/gene inference (e.g., 'global', 'experiment-wide', 'run-specific').
        glyco (bool): Flag indicating whether analysis is glycoform-specific.

    Methods:
        read(): Read data from the input file based on the alogorithm.
    """

    def __init__(self, config: RunnerIOConfig):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        infile = self.config.infile
        table = pd.read_csv(infile, sep="\t")
        return table


class TSVWriter(BaseWriter):
    """
    Class for writing OpenSWATH results to a tsv format.

    Attributes:
        infile (str): Input file path.
        outfile (str): Output file path.
        classifier (str): Classifier used for semi-supervised learning.
        level (str): Level used in semi-supervised learning (e.g., 'ms1', 'ms2', 'ms1ms2', 'transition', 'alignment').
        glyco (bool): Flag indicating whether analysis is glycoform-specific.

    Methods:
        save_results(result, pi0): Save the results to the output file based on the module using this class.
        save_weights(weights): Save the weights to the output file.
    """

    def __init__(self, config: RunnerIOConfig):
        super().__init__(config)

    def save_results(self, result, pi0):
        """
        Save the results of the scoring process to specified paths including summary statistics, final statistics, scored tables, and a report. The report includes various statistics such as cutoffs, svalues, qvalues, pvalues, and additional parameters like pi0 and color palette.
        """
        summ_stat_path = self.config.extra_writes.get("summ_stat_path")
        if summ_stat_path is not None:
            result.summary_statistics.to_csv(summ_stat_path, sep=",", index=False)
            logger.success("%s written." % summ_stat_path)

        full_stat_path = self.config.extra_writes.get("full_stat_path")
        if full_stat_path is not None:
            result.final_statistics.to_csv(full_stat_path, sep=",", index=False)
            logger.success("%s written." % full_stat_path)

        output_path = self.config.extra_writes.get("output_path")
        if output_path is not None:
            result.scored_tables.to_csv(output_path, sep="\t", index=False)
            logger.success("%s written." % output_path)

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[
                (result.scored_tables.peak_group_rank == 1)
                & (result.scored_tables.decoy == 0)
            ]["p_value"].values
            top_targets = result.scored_tables.loc[
                (result.scored_tables.peak_group_rank == 1)
                & (result.scored_tables.decoy == 0)
            ]["d_score"].values
            top_decoys = result.scored_tables.loc[
                (result.scored_tables.peak_group_rank == 1)
                & (result.scored_tables.decoy == 1)
            ]["d_score"].values

            save_report(
                self.config.extra_writes.get("report_path"),
                output_path,
                top_decoys,
                top_targets,
                cutoffs,
                svalues,
                qvalues,
                pvalues,
                pi0,
                self.config.runner.color_palette,
            )
            logger.success("%s written." % self.config.extra_writes.get("report_path"))
