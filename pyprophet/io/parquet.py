import pickle
from shutil import copyfile
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import click
from ._base import BaseReader, BaseWriter, BaseIOConfig
from .._config import RunnerIOConfig, IPFIOConfig, LevelContextIOConfig
from ..data_handling import get_parquet_column_names
from ..report import save_report
from ..glyco.report import save_report as save_report_glyco


class ParquetReader(BaseReader):
    """
    Class for reading and processing data from OpenSWATH results stored in Parquet format.

    The ParquetReader class provides methods to read different levels of data from the file and process it accordingly.
    It supports reading data for semi-supervised learning, IPF analysis, context level analysis.

    This assumes that the input file contains precursor and transition data.

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

    def read(self) -> pd.DataFrame:
        if isinstance(self.config, RunnerIOConfig):
            return self._read_for_semi_supervised()
        elif isinstance(self.config, IPFIOConfig):
            return self._read_for_ipf()
        elif isinstance(self.config, LevelContextIOConfig):
            return self._read_for_context_level()
        else:
            raise NotImplementedError(
                f"Unsupported config type: {type(self.config).__name__}"
            )

    def _read_for_semi_supervised(self) -> pd.DataFrame:
        ss_main_score = self.config.runner.ss_main_score

        all_column_names = get_parquet_column_names(self.infile)

        if self.level in ("ms2", "ms1ms2"):
            if not any([col.startswith("FEATURE_MS2_") for col in all_column_names]):
                raise click.ClickException(
                    "MS2-level feature columns are not present in parquet file."
                )

            cols = [
                "RUN_ID",
                "PRECURSOR_ID",
                "PRECURSOR_CHARGE",
                "FEATURE_ID",
                "EXP_RT",
                "PRECURSOR_DECOY",
                "TRANSITION_DETECTING",
            ]
            # Filter columes names for FEATURE_MS2_
            feature_ms2_cols = [
                col for col in all_column_names if col.startswith("FEATURE_MS2_")
            ]
            # Read the parquet file with selected columns
            table = pl.read_parquet(self.infile, columns=cols + feature_ms2_cols)

            # Drop rows with nulls in key columns
            table = table.drop_nulls(subset=["RUN_ID", "FEATURE_ID"] + feature_ms2_cols)

            # Rename columns
            table = table.rename(
                {
                    "PRECURSOR_DECOY": "DECOY",
                    **{
                        col: col.replace("FEATURE_MS2_", "")
                        for col in table.columns
                        if col.startswith("FEATURE_MS2_")
                    },
                }
            )

            # Create transition count table
            table_transition = (
                table.select(
                    [
                        "RUN_ID",
                        "PRECURSOR_ID",
                        "FEATURE_ID",
                        "EXP_RT",
                        "DECOY",
                        "TRANSITION_DETECTING",
                    ]
                )
                .explode("TRANSITION_DETECTING")
                .filter(pl.col("TRANSITION_DETECTING") == 1)
                .group_by(["RUN_ID", "PRECURSOR_ID", "FEATURE_ID", "EXP_RT", "DECOY"])
                .agg(pl.count().alias("TRANSITION_COUNT"))
            )

            # Join with main table
            table = table.join(
                table_transition,
                on=["RUN_ID", "PRECURSOR_ID", "FEATURE_ID", "EXP_RT", "DECOY"],
                how="left",
            ).drop("TRANSITION_DETECTING")

            # Create GROUP_ID column
            table = table.with_columns(
                (
                    pl.col("RUN_ID").cast(pl.Utf8)
                    + "_"
                    + pl.col("PRECURSOR_ID").cast(pl.Utf8)
                ).alias("GROUP_ID")
            )
            table = table.to_pandas()
        elif self.level == "ms1":
            if not any([col.startswith("FEATURE_MS1_") for col in all_column_names]):
                raise click.ClickException(
                    "MS1-level feature columns are not present in parquet file."
                )

            cols = [
                "RUN_ID",
                "PRECURSOR_ID",
                "PRECURSOR_CHARGE",
                "FEATURE_ID",
                "EXP_RT",
                "PRECURSOR_DECOY",
            ]
            # Filter columes names for FEATURE_MS1_
            feature_ms1_cols = [
                col for col in all_column_names if col.startswith("FEATURE_MS1_")
            ]
            # Read the parquet file with selected columns
            table = pl.read_parquet(self.infile, columns=cols + feature_ms1_cols)

            # Drop rows with nulls in key columns
            table = table.drop_nulls(subset=["RUN_ID", "FEATURE_ID"] + feature_ms1_cols)

            # Rename columns - remove 'FEATURE_MS1_' prefix and rename PRECURSOR_DECOY
            table = table.rename(
                {
                    "PRECURSOR_DECOY": "DECOY",
                    **{
                        col: col.replace("FEATURE_MS1_", "")
                        for col in table.columns
                        if col.startswith("FEATURE_MS1_")
                    },
                }
            )

            # Create GROUP_ID column by concatenating RUN_ID and PRECURSOR_ID
            table = table.with_columns(
                (
                    pl.col("RUN_ID").cast(pl.Utf8)
                    + "_"
                    + pl.col("PRECURSOR_ID").cast(pl.Utf8)
                ).alias("GROUP_ID")
            )
            table = table.to_pandas()
        elif self.level == "transition":
            if not any([col.startswith("SCORE_MS2_") for col in all_column_names]):
                raise click.ClickException(
                    "Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first."
                )
            if not any(
                [col.startswith("FEATURE_TRANSITION_") for col in all_column_names]
            ):
                raise click.ClickException(
                    "Transition-level feature columns are not present in parquet file."
                )

            cols = [
                "RUN_ID",
                "FEATURE_ID",
                "PRECURSOR_ID",
                "TRANSITION_ID",
                "PRECURSOR_DECOY",
                "TRANSITION_DECOY",
                "PRECURSOR_CHARGE",
                "TRANSITION_CHARGE",
            ]

            # Read only needed columns from parquet
            feature_transition_ms2_score_cols = [
                col
                for col in all_column_names
                if col.startswith("SCORE_MS2_") or col.startswith("FEATURE_TRANSITION_")
            ]

            table = pl.read_parquet(
                self.infile, columns=cols + feature_transition_ms2_score_cols
            )

            # Explode list columns
            exploding_cols = [
                col
                for col in table.columns
                if col.startswith("TRANSITION_")
                or col.startswith("FEATURE_TRANSITION_")
            ]

            table = table.explode(exploding_cols)

            # Apply filters
            table = table.filter(
                (pl.col("PRECURSOR_DECOY") == 0)
                & (
                    pl.col("SCORE_MS2_RANK")
                    <= self.config.runner.ipf_max_peakgroup_rank
                )
                & (pl.col("SCORE_MS2_PEP") <= self.config.runner.ipf_max_peakgroup_pep)
                & (
                    pl.col("FEATURE_TRANSITION_VAR_ISOTOPE_OVERLAP_SCORE")
                    <= self.config.runner.ipf_max_transition_isotope_overlap
                )
                & (
                    pl.col("FEATURE_TRANSITION_VAR_LOG_SN_SCORE")
                    > self.config.runner.ipf_min_transition_sn
                )
            )

            # Drop SCORE_MS2_ columns
            score_cols = [col for col in table.columns if col.startswith("SCORE_MS2_")]
            table = table.drop(score_cols)

            # Rename columns
            table = table.rename(
                {
                    "TRANSITION_DECOY": "DECOY",
                    **{
                        col: col.replace("FEATURE_TRANSITION_", "")
                        for col in table.columns
                        if col.startswith("FEATURE_TRANSITION_")
                    },
                }
            )

            # Create GROUP_ID column
            table = table.with_columns(
                (
                    pl.col("RUN_ID").cast(pl.Utf8)
                    + "_"
                    + pl.col("FEATURE_ID").cast(pl.Utf8)
                    + "_"
                    + pl.col("PRECURSOR_ID").cast(pl.Utf8)
                    + "_"
                    + pl.col("TRANSITION_ID").cast(pl.Utf8)
                ).alias("GROUP_ID")
            )
            table = table.to_pandas()
        else:
            raise click.ClickException("Unspecified data level selected.")

        # Append MS1 scores to MS2 table if selected
        if self.level == "ms1ms2":
            if not any([col.startswith("FEATURE_MS1_") for col in all_column_names]):
                raise click.ClickException(
                    "MS1-level feature columns are not present in parquet file."
                )

            # Filter columes names for FEATURE_MS1_
            feature_ms1_cols = [
                col for col in all_column_names if col.startswith("FEATURE_MS1_VAR")
            ]
            # Read the parquet file
            ms1_table = pd.read_parquet(
                self.infile, columns=["FEATURE_ID"] + feature_ms1_cols
            )
            # Rename columns with 'FEATURE_MS1_VAR_' to 'VAR_MS1_'
            ms1_table.columns = [
                col.replace("FEATURE_MS1_VAR_", "VAR_MS1_") for col in ms1_table.columns
            ]

            table = pd.merge(table, ms1_table, how="left", on="FEATURE_ID")

        # Format table
        table.columns = [col.lower() for col in table.columns]

        # Mark main score column
        if ss_main_score.lower() in table.columns:
            table = table.rename(
                index=str,
                columns={ss_main_score.lower(): "main_" + ss_main_score.lower()},
            )
        elif ss_main_score.lower() == "swath_pretrained":
            # Add a pretrained main score corresponding to the original implementation in OpenSWATH
            # This is optimized for 32-windows SCIEX TripleTOF 5600 data
            table["main_var_pretrained"] = -(
                -0.19011762 * table["var_library_corr"]
                + 2.47298914 * table["var_library_rmsd"]
                + 5.63906731 * table["var_norm_rt_score"]
                + -0.62640133 * table["var_isotope_correlation_score"]
                + 0.36006925 * table["var_isotope_overlap_score"]
                + 0.08814003 * table["var_massdev_score"]
                + 0.13978311 * table["var_xcorr_coelution"]
                + -1.16475032 * table["var_xcorr_shape"]
                + -0.19267813 * table["var_yseries_score"]
                + -0.61712054 * table["var_log_sn_score"]
            )
        else:
            raise click.ClickException(
                f"Main score ({ss_main_score.lower()}) column not present in data. Current columns: {table.columns}"
            )

        # Enable transition count & precursor / product charge scores for XGBoost-based classifier
        if self.classifier == "XGBoost" and self.level != "alignment":
            click.echo(
                "Info: Enable number of transitions & precursor / product charge scores for XGBoost-based classifier"
            )
            table = table.rename(
                index=str,
                columns={
                    "precursor_charge": "var_precursor_charge",
                    "product_charge": "var_product_charge",
                    "transition_count": "var_transition_count",
                },
            )

        return table

    def _read_for_ipf(self):
        # implement logic from `ipf.py`
        raise NotImplementedError

    def _read_for_context_level(self):
        # implement logic from `levels_context.py`
        raise NotImplementedError


class ParquetWriter(BaseWriter):
    """
    Class for writing OpenSWATH results to a Parquet file.

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

    def save_results(self, result, pi0):
        if isinstance(self.config, RunnerIOConfig):
            return self._save_semi_supervised_results(result, pi0)
        elif isinstance(self.config, IPFIOConfig):
            return self._save_ipf_results(result)
        elif isinstance(self.config, LevelContextIOConfig):
            return self._save_context_level_results(result)
        else:
            raise NotImplementedError(
                f"Unsupported config type: {type(self.config).__name__}"
            )

    def _save_semi_supervised_results(self, result, pi0):
        # extract logic from runner.py save_osw_results
        raise NotImplementedError

    def _save_ipf_results(self, result):
        # extract logic from ipf.py
        raise NotImplementedError

    def _save_context_level_results(self, result):
        # extract logic from levels_context.py
        raise NotImplementedError
