"""
This module provides abstract base classes and utility classes for reading and writing
data in various formats (e.g., OSW, Parquet, Split Parquet) used in the OpenSWATH workflow.

The module defines the following:
- Abstract base classes (`BaseReader`, `BaseWriter`) for implementing custom readers and writers.
- Concrete base classes for specific file formats, such as OSW and Parquet.
- Utility methods for handling data validation, column preparation, and file merging.

Classes:
    - RowCountMismatchError: Custom exception for row count mismatches during data processing.
    - BaseReader: Abstract base class for implementing readers for different data formats.
    - BaseWriter: Abstract base class for implementing writers for different data formats.
    - BaseOSWReader: Reader class for OSW (SQLite-based) files.
    - BaseOSWWriter: Writer class for OSW (SQLite-based) files.
    - BaseParquetReader: Reader class for Parquet files.
    - BaseParquetWriter: Writer class for Parquet files.
    - BaseSplitParquetReader: Reader class for split Parquet files (multi-run support).
    - BaseSplitParquetWriter: Writer class for split Parquet files (multi-run support).

Key Features:
    - Abstract methods (`read`, `save_results`) enforce implementation in subclasses.
    - Support for semi-supervised learning, IPF analysis, and context-level inference.
    - Validation of row counts to ensure data consistency during processing.
    - Utility methods for handling Parquet-specific operations, such as collapsing protein IDs.

Usage:
    This module is intended to be extended by specific implementations for reading and writing
    data in the OpenSWATH workflow. Subclasses must implement the abstract methods to handle
    specific file formats and processing requirements.

Dependencies:
    - pandas
    - polars
    - duckdb
    - click
    - loguru
"""

import glob
import os
import pickle
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import click
import duckdb
import pandas as pd
import polars as pl
from loguru import logger

from .._base import BaseIOConfig
from .._config import ExportIOConfig
from ..report import save_report
from .util import get_parquet_column_names, print_parquet_tree


class RowCountMismatchError(Exception):
    """
    Exception to raise when the new output with joined scores has a different amount of rows than the input.
    """

    pass


@dataclass
class BaseReader(ABC):
    """
    Abstract base class for implementing readers that load data from different sources (OSW, Parquet, etc.).
    """

    def __init__(self, config: BaseIOConfig):
        """
        Initialize the reader with a given configuration.

        Args:
            config (BaseIOConfig): Configuration object containing input details, and module specific config for params for reading.
        """
        self.config = config

    @property
    def context(self):
        return self.config.context

    @property
    def infile(self):
        return self.config.infile

    @property
    def outfile(self):
        return self.config.outfile

    @property
    def file_type(self):
        return self.config.file_type

    @property
    def subsample_ratio(self):
        return self.config.subsample_ratio

    @property
    def classifier(self):
        return self.config.runner.classifier

    @property
    def level(self):
        return self.config.level

    @property
    def glyco(self):
        return self.config.runner.glyco

    @abstractmethod
    def read(self):
        """
        Abstract method to be implemented by subclasses to read data from a specific format.
        """
        raise NotImplementedError("Subclasses must implement 'read'.")

    # ----------------------------
    # Shared Scoring Methods
    # ----------------------------

    def _finalize_feature_table(self, df, ss_main_score):
        """
        Finalize the feature table for semi-supervised scoring.
        """
        df.columns = [c.lower() for c in df.columns]
        main_score = ss_main_score.lower()
        if main_score in df.columns:
            df = df.rename(columns={main_score: "main_" + main_score})
        elif (
            main_score == "swath_pretrained"
        ):  # TODO: Should we deprecate this? This is probably not optimal for newer data from different instruments
            # Add a pretrained main score corresponding to the original implementation in OpenSWATH
            # This is optimized for 32-windows SCIEX TripleTOF 5600 data
            df["main_var_pretrained"] = -(
                -0.19011762 * df["var_library_corr"]
                + 2.47298914 * df["var_library_rmsd"]
                + 5.63906731 * df["var_norm_rt_score"]
                + -0.62640133 * df["var_isotope_correlation_score"]
                + 0.36006925 * df["var_isotope_overlap_score"]
                + 0.08814003 * df["var_massdev_score"]
                + 0.13978311 * df["var_xcorr_coelution"]
                + -1.16475032 * df["var_xcorr_shape"]
                + -0.19267813 * df["var_yseries_score"]
                + -0.61712054 * df["var_log_sn_score"]
            )
        else:
            raise click.ClickException(
                f"Main score ({main_score}) not found in input columns: {df.columns}"
            )

        if self.classifier == "XGBoost" and self.level != "alignment":
            logger.info(
                "Enable number of transitions & precursor / product charge scores for XGBoost-based classifier"
            )
            df = df.rename(
                columns={
                    "precursor_charge": "var_precursor_charge",
                    "product_charge": "var_product_charge",
                    "transition_count": "var_transition_count",
                }
            )

        # print(df.info())
        return df


@dataclass
class BaseWriter(ABC):
    """
    Abstract base class for implementing writers that save results to various output formats.
    """

    def __init__(self, config: BaseIOConfig):
        """
        Initialize the writer with a given configuration.

        Args:
            config (BaseIOConfig): Configuration object containing output details.
        """
        self.config = config
        self.__post_init__()

    def __post_init__(self):
        """
        Post-initialization method to set up variables for IO specific config
        """
        if isinstance(self.config, ExportIOConfig):
            # Quantification matrix normalization methods
            self.normalization_methods = {
                "median": self._median_normalize,
                "medianmedian": self._median_median_normalize,
                "quantile": self._quantile_normalize,
                "none": lambda x: x,
            }

    @property
    def context(self):
        return self.config.context

    @property
    def infile(self):
        return self.config.infile

    @property
    def outfile(self):
        return self.config.outfile

    @property
    def file_type(self):
        return self.config.file_type

    @property
    def classifier(self):
        return self.config.runner.classifier

    @property
    def level(self):
        return self.config.level

    @property
    def glyco(self):
        return self.config.runner.glyco

    # ----------------------------
    # Shared Common Methods
    # ----------------------------

    @abstractmethod
    def save_results(self, result, pi0):
        """
        Abstract method to save scoring results and statistical outputs.

        Args:
            result: The result object containing scoring tables.
            pi0: Estimated pi0 value from FDR statistics.
        """
        raise NotImplementedError("Subclasses must implement 'save_results'.")

    def _get_columns_to_keep(
        self, existing_cols: list[str], score_prefix: str
    ) -> list[str]:
        """
        Get the columns to keep in the DataFrame by removing existing score columns. Mainly for Parquet files.

        Note: this method itself does not remove the columns, it just returns a list of columns to keep that does not include the existing score columns.
        """
        drop_cols = [col for col in existing_cols if col.startswith(score_prefix)]
        if drop_cols:
            logger.warning(
                f"Warn: Dropping existing {score_prefix} columns.",
            )

        return [col for col in existing_cols if not col.startswith(score_prefix)]

    def _get_parquet_row_count(self, con, target_file: str) -> int:
        """
        Get the row count of a Parquet file.

        Parameters:
            con: DuckDB connection object
            target_file: Path to the Parquet file

        Returns:
            int: Row count of the Parquet file
        """
        query = f"SELECT COUNT(*) FROM read_parquet('{target_file}')"
        row_count = con.execute(query).fetchone()[0]
        return row_count

    def _validate_row_count_after_join(
        self, con, target_file: str, key_cols: str, join_on: str, prefix: str
    ):
        """
        Validates the row count after performing a join operation on a Parquet file.

        This is important, because we would not expect the appending of scores to change the number of rows in the input Parquet file.

        Parameters:
            con: DuckDB connection object
            target_file: Path to the Parquet file
            key_cols: The key columns for the join operation
            join_on: The condition for the join operation
            prefix: The prefix (table alias) used for the Parquet file in the query

        Raises:
            RowCountMismatchError: If the row count of the resulting join doesn't match the original row count.
        """

        original_row_count = self._get_parquet_row_count(con, target_file)

        logger.debug(
            f"Prior to appending scores, {target_file} has {original_row_count} entries"
        )

        validate_query = f"""
            SELECT 
                COUNT(*) AS row_count
            FROM (
                SELECT {key_cols}
                FROM read_parquet('{target_file}') {prefix}
                LEFT JOIN scores s
                ON {join_on}
            ) AS subquery
        """

        logger.trace(f"Row Entry validation query:\n{validate_query}")

        result = con.execute(validate_query).fetchone()
        resulting_row_count = result[0]

        # If row count mismatch occurs, raise an error and log it
        if resulting_row_count != original_row_count:
            error_message = (
                f"There was an issue with appending scores to {target_file}.\n"
                f"Row count mismatch: Original rows {original_row_count} ≠ Resulting rows {resulting_row_count}.\n"
                "Appending scores resulted in a different number of entries than the original input file. This is unexpected behaviour."
            )

            try:
                raise RowCountMismatchError(error_message)
            except RowCountMismatchError as e:
                logger.opt(exception=e, depth=1).critical(f"Critical error: {str(e)}")

            sys.exit(1)

    # ----------------------------
    # Shared Scoring Methods
    # ----------------------------

    def save_weights(self, weights):
        """
        Abstract method to save model weights (e.g., LDA coefficients, XGBoost model).

        Args:
            weights: Model weights or trained object.
        """
        if self.classifier == "LDA":
            self._save_tsv_weights(weights)
        elif self.classifier == "SVM":
            self._save_tsv_weights(weights)
        elif self.classifier == "XGBoost":
            self._save_bin_weights(weights)
        else:
            raise NotImplementedError(
                f"Classifier {self.classifier} not supported for saving weights."
            )

    def _prepare_score_dataframe(
        self, df: pd.DataFrame, level: str, prefix: str
    ) -> pd.DataFrame:
        """
        Prepare the score DataFrame
        """
        score_cols = [
            "feature_id",
            "d_score",
            "peak_group_rank",
            "p_value",
            "q_value",
            "pep",
        ]
        if "h_score" in df.columns:
            score_cols.insert(2, "h_score")
            score_cols.insert(3, "h0_score")

        if level in ("ms1", "ms2", "ms1ms2") and self.config.file_type in (
            "parquet",
            "parquet_split",
            "parquet_split_multi",
        ):
            # For parquet files, there may be multiple proteins that map to the same peptide
            score_cols.insert(2, "protein_id")

            # # Check if protein_id column is collapsed, if it is, we need to explode it
            # if (
            #     df["protein_id"]
            #     .apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 1)
            #     .any()
            # ):
            #     logger.info(
            #         "Exploding protein_id column to handle multiple proteins per feature."
            #     )
            df = df.explode("protein_id")

        if level == "transition":
            if self.file_type in ("parquet", "parquet_split", "parquet_split_multi"):
                score_cols.insert(1, "ipf_peptide_id")
                score_cols.insert(2, "transition_id")

                # For parquet files, there may be multiple ipf_peptide_ids that map to the same transition feature
                df = df.explode("ipf_peptide_id")
            else:
                score_cols.insert(1, "transition_id")

        if level == "alignment":
            score_cols.insert(2, "alignment_id")
            score_cols.insert(2, "decoy")
            # Reverse map DECOY 0 to 1 and 1 to -1
            # arycal saves a label column to indicate 1 as target aligned peaks and -1 as the random/shuffled decoy aligned peak
            df["decoy"] = df["decoy"].map({0: 1, 1: -1})

        df = df[score_cols].rename(columns=str.upper)
        df = df.rename(columns={"D_SCORE": "SCORE"})

        if self.config.file_type == "osw":
            # Consistent renaming with original implementation for OSW files
            df = df.rename(columns={"P_VALUE": "PVALUE", "Q_VALUE": "QVALUE"})

        if self.config.file_type == "osw" or self.level not in ("ms1ms2", "ms2"):
            df = df.rename(columns={"PEAK_GROUP_RANK": "RANK"})

        if level == "transition":
            if self.file_type in ("parquet", "parquet_split", "parquet_split_multi"):
                key_cols = {"FEATURE_ID", "IPF_PEPTIDE_ID", "TRANSITION_ID"}
            else:
                key_cols = {"FEATURE_ID", "TRANSITION_ID"}
        elif level == "alignment":
            key_cols = {"ALIGNMENT_ID", "FEATURE_ID", "DECOY"}
        elif level in ("ms1", "ms2", "ms1ms2") and self.config.file_type in (
            "parquet",
            "parquet_split",
            "parquet_split_multi",
        ):
            key_cols = {"FEATURE_ID", "PROTEIN_ID"}
        else:
            key_cols = {"FEATURE_ID"}

        if self.config.file_type == "osw":
            return df

        rename_map = {
            col: f"{prefix}{col}" for col in df.columns if col not in key_cols
        }
        return df.rename(columns=rename_map)

    def _write_pdf_report(self, result, pi0):
        """
        Write a PDF report if the scoring results contain final statistics.
        """

        if result.final_statistics is None:
            return

        df = result.scored_tables
        if self.level == "alignment":
            # Map Decoy 1 to 0 and -1 to 1
            df["decoy"] = df["decoy"].map({1: 0, -1: 1})

        # print(df)
        # print(df.columns)

        prefix = self.config.prefix
        level = self.level

        cutoffs = result.final_statistics["cutoff"].values
        svalues = result.final_statistics["svalue"].values
        qvalues = result.final_statistics["qvalue"].values

        pvalues = df[(df.peak_group_rank == 1) & (df.decoy == 0)]["p_value"].values
        top_targets = df[(df.peak_group_rank == 1) & (df.decoy == 0)]["d_score"].values
        top_decoys = df[(df.peak_group_rank == 1) & (df.decoy == 1)]["d_score"].values

        # Check if any of the values are empty, can't create a report if they are
        if not all(
            [
                len(top_targets),
                len(top_decoys),
                len(cutoffs),
                len(svalues),
                len(qvalues),
                len(pvalues),
            ]
        ):
            logger.error("Not enough values to create a report.")
            logger.error(f"top_targets: {len(top_targets)}")
            logger.error(f"top_decoys: {len(top_decoys)}")
            logger.error(f"cutoffs: {len(cutoffs)}")
            logger.error(f"svalues: {len(svalues)}")
            logger.error(f"qvalues: {len(qvalues)}")
            logger.error(f"pvalues: {len(pvalues)}")
            logger.error(f"pi0: {len(pi0)}")
            return

        pdf_path = os.path.join(prefix + f"_{level}_report.pdf")
        save_report(
            pdf_path,
            self.outfile,
            top_decoys,
            top_targets,
            cutoffs,
            svalues,
            qvalues,
            pvalues,
            pi0,
            self.config.runner.color_palette,
            self.level,
            df=df,
        )
        logger.success(f"{pdf_path} written.")

    def _save_tsv_weights(self, weights):
        """
        Save the model weights to a TSV file, ensuring no duplicate levels.

        If weights for the current level already exist, they are removed before saving the new ones.
        """
        weights["level"] = self.level
        trained_weights_path = self.config.extra_writes.get("trained_weights_path")

        if trained_weights_path is not None:
            if os.path.exists(trained_weights_path):
                existing_df = pd.read_csv(trained_weights_path, sep=",")
                existing_df = existing_df[existing_df["level"] != self.level]
                updated_df = pd.concat([existing_df, weights], ignore_index=True)
            else:
                updated_df = weights

            # Always overwrite with a single header
            updated_df.to_csv(trained_weights_path, sep=",", index=False)
            logger.success(f"{trained_weights_path} written.")

    def _save_bin_weights(self, weights):
        """
        Save the model weights to a binary file.

        Args:
            weights: Model weights or trained object.
        """
        trained_weights_path = self.config.extra_writes.get(
            f"trained_model_path_{self.level}"
        )
        if trained_weights_path is not None:
            with open(trained_weights_path, "wb") as file:
                self.persisted_weights = pickle.dump(weights, file)
            logger.success("%s written." % trained_weights_path)
        else:
            logger.error(f"Trained model path {trained_weights_path} not found. ")

    # ----------------------------
    # Shared Levels Context Methods
    # ----------------------------

    def _write_levels_context_pdf_report(self, data, stat_table, pi0):
        """
        Write a PDF report for levels context.
        """
        context = self.config.context_fdr
        prefix = self.config.prefix
        analyte = self.config.level
        if context == "run-specific":
            prefix = prefix + "_" + str(data["run_id"].unique()[0])

        pdf_path = os.path.join(prefix + "_" + context + "_" + analyte + "_report.pdf")
        title = prefix + "_" + context + "_" + analyte + "-level error-rate control"
        save_report(
            pdf_path,
            title,
            data[data.decoy == 1]["score"].values,
            data[data.decoy == 0]["score"].values,
            stat_table["cutoff"].values,
            stat_table["svalue"].values,
            stat_table["qvalue"].values,
            data[data.decoy == 0]["p_value"].values,
            pi0,
            self.config.color_palette,
            analyte,
            data,
        )
        logger.success(f"{pdf_path} written.")

    # ----------------------------
    # Shared Export Methods
    # ----------------------------

    def export_results(self, data: pd.DataFrame):
        """
        Save the results to the output file based on the export format.

        Args:
            data: DataFrame containing the data to be exported
        """
        cfg = self.config

        sep = "," if cfg.out_type == "csv" else "\t"

        if cfg.export_format == "legacy_split":
            logger.info("Exporting results in legacy split format.")
            data = data.drop(["id_run", "id_peptide"], axis=1)
            # filename might contain archive extensions, so we need to remove these
            data["filename"] = data["filename"].apply(
                lambda x: os.path.splitext(os.path.basename(x))[0]
            )
            data.groupby("filename").apply(
                lambda x: (
                    x.to_csv(
                        os.path.basename(x["filename"].values[0]) + f".{cfg.out_type}",
                        sep=sep,
                        index=False,
                    ),
                    logger.success(
                        f"Exported results to {os.path.basename(x['filename'].values[0])}.{cfg.out_type}."
                    ),
                )
            )
        elif cfg.export_format == "legacy_merged":
            logger.info(
                f"Exporting results ({data.shape} | {sys.getsizeof(data) / (1024**2):.2f}MB ) in legacy merged format."
            )
            data.drop(["id_run", "id_peptide"], axis=1).to_csv(
                cfg.outfile, sep=sep, index=False
            )
            logger.success(f"Exported results to {cfg.outfile}.")
        else:
            raise ValueError(f"Unsupported export format: {cfg.export_format}")

    def export_quant_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Export quantification matrix at specified level with optional normalization.

        Args:
            data: Input DataFrame with quantification data

        """
        cfg = self.config

        sep = "," if cfg.out_type == "csv" else "\t"
        level = self.level
        normalization = self.config.normalization
        # Validate input
        if level not in ["precursor", "peptide", "protein", "gene"]:
            raise ValueError(
                "Invalid level. Choose from: precursor, peptide, protein, gene"
            )

        if normalization not in self.normalization_methods:
            raise ValueError(
                f"Invalid normalization. Choose from: {list(self.normalization_methods.keys())}"
            )

        # Get the appropriate summarization method
        summarizer = getattr(self, f"_summarize_{level}_level")
        matrix = summarizer(data, self.config.top_n, self.config.consistent_top)

        # Apply normalization
        if normalization != "none":
            # Set non-numeric columns as index
            non_numeric_cols = list(matrix.select_dtypes(exclude="number").columns)

            if len(non_numeric_cols) > 0:
                matrix = matrix.set_index(non_numeric_cols)
            matrix = self.normalization_methods[normalization](matrix)
            matrix = matrix.reset_index()

        matrix.to_csv(self.config.outfile, sep=sep, index=False)

    def _summarize_precursor_level(
        self, data: pd.DataFrame, _top_n: int, _consistent_top: bool
    ) -> pd.DataFrame:
        """
        Create precursor-level matrix (no summarization needed).
        Just select top peak group per precursor.
        """
        # Select top ranking peak group only
        data = data.iloc[
            data.groupby(["run_id", "transition_group_id"]).apply(
                lambda x: x["m_score"].idxmin()
            )
        ]
        logger.info("Summarizing to precursor level.")
        # Create matrix
        matrix = data.pivot_table(
            index=[
                "transition_group_id",
                "Sequence",
                "FullPeptideName",
                "Charge",
                "ProteinName",
            ],
            columns="filename",
            values="Intensity",
        ).reset_index()
        return matrix

    def _summarize_peptide_level(
        self, data: pd.DataFrame, top_n: int, consistent_top: bool
    ) -> pd.DataFrame:
        """
        Summarize to peptide level using top N precursors.
        """
        # First get top peak group per precursor
        data = data.iloc[
            data.groupby(["run_id", "transition_group_id"]).apply(
                lambda x: x["m_score"].idxmin()
            )
        ]
        logger.info("Summarizing to peptide level.")
        # Get top precursors for each peptide
        if consistent_top:
            logger.info("Using consistent top precursors across all runs.")
            # Use precursors with highest median intensity across all runs
            median_intensity = (
                data.groupby(["transition_group_id", "Sequence", "FullPeptideName"])[
                    "Intensity"
                ]
                .median()
                .reset_index()
            )

            top_precursors = (
                median_intensity.groupby(["Sequence", "FullPeptideName"])
                .apply(lambda x: x.nlargest(top_n, "Intensity")["transition_group_id"])
                .reset_index()["transition_group_id"]
            )

            data = data[data["transition_group_id"].isin(top_precursors)]
        else:
            logger.info("Using top precursors per run individually.")
            # Select top precursors per run individually
            data = (
                data.groupby(["run_id", "Sequence", "FullPeptideName"])
                .apply(lambda x: x.nlargest(top_n, "Intensity"))
                .reset_index(drop=True)
            )

        # Summarize by peptide (mean of top precursors)
        peptide_matrix = (
            data.groupby(["Sequence", "FullPeptideName", "filename"])["Intensity"]
            .mean()
            .unstack()
        ).reset_index()
        return peptide_matrix

    def _summarize_protein_level(
        self, data: pd.DataFrame, top_n: int, consistent_top: bool
    ) -> pd.DataFrame:
        """
        Summarize to protein level using top N peptides.
        """
        # First summarize to peptide level
        peptide_matrix = self._summarize_peptide_level(
            data, top_n=top_n, consistent_top=consistent_top
        )

        logger.info("Summarizing to protein level.")

        # Need to get protein annotations - get from original data
        protein_map = data.drop_duplicates(
            ["FullPeptideName", "ProteinName"]
        ).set_index("FullPeptideName")["ProteinName"]

        # Split protein groups and explode (one row per protein)
        protein_matrix = peptide_matrix.copy()
        protein_matrix = protein_matrix.merge(
            protein_map.reset_index(),
            left_on="FullPeptideName",
            right_on="FullPeptideName",
            how="left",
        )
        protein_matrix = protein_matrix.explode("ProteinName")

        if consistent_top:
            # Calculate median intensity for each peptide-protein combination
            protein_matrix["median_intensity"] = protein_matrix.select_dtypes(
                include="number"
            ).median(axis=1)

            # Get top N peptides per protein based on median intensity
            top_peptides = protein_matrix.groupby("ProteinName").apply(
                lambda x: x.nlargest(top_n, "median_intensity")
            )

            # Flatten the multi-index and select the top peptides
            top_peptides = top_peptides.droplevel(0)
            protein_matrix = protein_matrix.loc[top_peptides.index]

            # Drop the median_intensity column as it is no longer needed
            protein_matrix = protein_matrix.drop(columns=["median_intensity"])

        # Summarize by protein (mean of top peptides)
        protein_matrix = (
            protein_matrix.groupby("ProteinName").mean(numeric_only=True).reset_index()
        )

        return protein_matrix

    def _summarize_gene_level(
        self, data: pd.DataFrame, top_n: int, consistent_top: bool
    ) -> pd.DataFrame:
        """
        Summarize to gene level using top N peptides.
        """
        # First summarize to peptide level
        peptide_matrix = self._summarize_peptide_level(
            data, top_n=top_n, consistent_top=consistent_top
        )

        logger.info("Summarizing to gene level.")

        # Need to get gene annotations - get from original data
        gene_map = data.drop_duplicates(["FullPeptideName", "Gene"]).set_index(
            "FullPeptideName"
        )["Gene"]

        # Split gene groups and explode (one row per gene)
        gene_matrix = peptide_matrix.copy()
        gene_matrix = gene_matrix.merge(
            gene_map.reset_index(),
            left_on="FullPeptideName",
            right_on="FullPeptideName",
            how="left",
        )
        gene_matrix = gene_matrix.explode("Gene")

        if consistent_top:
            # Calculate median intensity for each peptide-gene combination
            gene_matrix["median_intensity"] = gene_matrix.select_dtypes(
                include="number"
            ).median(axis=1)

            # Get top N peptides per gene based on median intensity
            top_peptides = gene_matrix.groupby("Gene").apply(
                lambda x: x.nlargest(top_n, "median_intensity")
            )

            # Flatten the multi-index and select the top peptides
            top_peptides = top_peptides.droplevel(0)
            gene_matrix = gene_matrix.loc[top_peptides.index]

            # Drop the median_intensity column as it is no longer needed
            gene_matrix = gene_matrix.drop(columns=["median_intensity"])

        # Summarize by gene (mean of top peptides)
        gene_matrix = gene_matrix.groupby("Gene").mean(numeric_only=True).reset_index()

        return gene_matrix

    def _median_normalize(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Median normalization (per sample)"""
        logger.info("Applying median normalization.")
        return matrix.div(matrix.median(axis=0), axis=1)

    def _median_median_normalize(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Median of medians normalization"""
        logger.info("Applying median of medians normalization.")
        sample_medians = matrix.median(axis=0)
        global_median = sample_medians.median()
        return matrix.div(sample_medians, axis=1) * global_median

    def _quantile_normalize(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Quantile normalization"""
        try:
            from sklearn.preprocessing import quantile_transform

            logger.info("Applying quantile normalization.")
            # Transpose to normalize samples (columns) together
            normalized = quantile_transform(matrix.T, copy=True).T
            return pd.DataFrame(normalized, index=matrix.index, columns=matrix.columns)
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for quantile normalization"
            ) from exc


@dataclass
class BaseOSWReader(BaseReader):
    """
    Class for reading and processing data from an OpenSWATH workflow OSW-sqlite based file.

    The OSWReader class provides methods to read different levels of data from the file and process it accordingly.
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

    def read(self) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses to read data from OSW format for a specific algorithm.
        """
        raise NotImplementedError(
            "The read method must be implemented in subclasses of BaseOSWReader."
        )


@dataclass
class BaseOSWWriter(BaseWriter):
    """
    Class for writing OpenSWATH results to an OSW-sqlite based file.

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
        """
        Abstract method to save scoring results and statistical outputs.
        """
        raise NotImplementedError(
            "The save_results method must be implemented in subclasses of BaseOSWWriter."
        )


@dataclass
class BaseParquetReader(BaseReader):
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

        if self.config.context == "ipf" and config.propagate_signal_across_runs:
            # We make the assumption that the alignment file is in the same directory as the input file
            self.alignment_file = os.path.join(
                os.path.dirname(self.infile), "feature_alignment.parquet"
            )
            if not os.path.exists(self.alignment_file):
                raise click.ClickException(
                    f"To use the --propagate-signal-across-runs option, "
                    f"the alignment file {self.alignment_file} must exist."
                )

    def read(self) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses to read data from Parquet format for a specific algorithm.
        """
        raise NotImplementedError(
            "The read method must be implemented in subclasses of BaseParquetReader."
        )

    def _init_duckdb_views(self, con):
        # Create TEMP table of sampled precursors IDs (if needed)
        if self.subsample_ratio < 1.0:
            logger.info(
                f"Subsampling {self.subsample_ratio * 100}% data for semi-supervised learning."
            )
            con.execute(
                f"""
                CREATE TEMP TABLE sampled_precursor_ids AS
                SELECT DISTINCT PRECURSOR_ID
                FROM read_parquet('{self.infile}')
                USING SAMPLE {self.subsample_ratio * 100}%
                """
            )
            n = con.execute("SELECT COUNT(*) FROM sampled_precursor_ids").fetchone()[0]
            logger.info(f"Sampled {n} precursor IDs")

            con.execute(
                f"CREATE VIEW data AS SELECT * FROM read_parquet('{self.infile}') WHERE PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sampled_precursor_ids)"
            )
        else:
            con.execute(
                f"CREATE VIEW data AS SELECT * FROM read_parquet('{self.infile}')"
            )

        if self.context == "ipf" and self.config.propagate_signal_across_runs:
            con.execute(
                f"CREATE VIEW alignment AS SELECT * FROM read_parquet('{self.alignment_file}')"
            )

    def _get_columns_by_prefix(self, parquet_file, prefix):
        cols = get_parquet_column_names(parquet_file)
        return [c for c in cols if c.startswith(prefix)]

    def _collapse_protein_ids(self, df):
        """
        Collapse protein IDs to avoid duplicating feature data in the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame containing feature data.

        Returns:
        - DataFrame: DataFrame with collapsed protein IDs.

        Example:
        df = _collapse_protein_ids(df)
        """
        org_shape = df.shape
        # Handle cases where there may be multiple proteins mapping to a peptide with the same feature ID.
        # We collapse the protein IDs to avoid duplicating feature data
        df = (
            df.group_by(["GROUP_ID", "FEATURE_ID"])
            .agg(
                [
                    pl.col("PROTEIN_ID").unique().alias("PROTEIN_ID"),
                    *[
                        pl.col(c).first()
                        for c in df.columns
                        if c not in {"GROUP_ID", "FEATURE_ID", "PROTEIN_ID"}
                    ],
                ]
            )
            .sort(["RUN_ID", "PRECURSOR_ID", "EXP_RT"])
        )
        collapsed_shape = df.shape

        if org_shape != collapsed_shape:
            logger.info(
                f"Collapsed {org_shape[0] - collapsed_shape[0]} of {org_shape[0]} rows due to multiple proteins mapping to the same feature ID."
            )

        # print(
        #     df.filter(
        #         pl.col("PRECURSOR_ID").is_in([7012])
        #         & pl.col("FEATURE_ID").is_in([33731659783088])
        #     )
        # )
        return df

    def _collapse_ipf_peptide_ids(self, df):
        """
        Collapse IPF peptide IDs to avoid duplicating feature data in the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame containing feature data.

        Returns:
        - DataFrame: DataFrame with collapsed IPF peptide IDs.
        """
        org_shape = df.shape
        # Handle cases where there may be multiple peptides mapping to a precursor with the same feature ID.
        # We collapse the IPF peptide IDs to avoid duplicating feature data
        df = (
            df.group_by(["GROUP_ID", "FEATURE_ID"])
            .agg(
                [
                    pl.col("IPF_PEPTIDE_ID").unique().alias("IPF_PEPTIDE_ID"),
                    *[
                        pl.col(c).first()
                        for c in df.columns
                        if c not in {"GROUP_ID", "FEATURE_ID", "IPF_PEPTIDE_ID"}
                    ],
                ]
            )
            .sort(["RUN_ID", "TRANSITION_ID"])
        )
        collapsed_shape = df.shape

        if org_shape != collapsed_shape:
            logger.info(
                f"Collapsed {org_shape[0] - collapsed_shape[0]} of {org_shape[0]} rows due to multiple transitions features mapping to different peptidoforms."
            )

        return df


@dataclass
class BaseParquetWriter(BaseWriter):
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

        if self.context == "levels_context":
            self.context_level_id_map = {
                "peptide": "peptide_id",
                "protein": "protein_id",
                "gene": "gene_id",
            }

    def save_results(self, result, pi0):
        """
        Abstract method to save scoring results and statistical outputs.
        """
        raise NotImplementedError(
            "The save_results method must be implemented in subclasses of ParquetWriter."
        )


@dataclass
class BaseSplitParquetReader(BaseParquetReader):
    """
    Class for reading and processing data from OpenSWATH results stored in a directoy containing split Parquet files.

    The ParquetReader class provides methods to read different levels of data from the split parquet files and process it accordingly.
    It supports reading data for semi-supervised learning, IPF analysis, context level analysis.

    This assumes that the input infile path is a directory containing the following files:
    - precursors_features.parquet
    - transition_features.parquet
    - feature_alignment.parquet (optional)

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
        self.config = config

        if config.file_type not in ("parquet_split", "parquet_split_multi"):
            raise click.ClickException(
                f"SplitParquetReader requires 'parquet_split' or 'parquet_split_multi' input, got '{config.file_type}' instead."
            )

        # Flag to indicate whether the input is a multi-run directory
        self._is_multi_run = config.file_type == "parquet_split_multi"

    def read(self) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses to read data from splti parquet format for a specific algorithm.
        """
        raise NotImplementedError(
            "The read method must be implemented in subclasses of BaseSplitParquetReader."
        )

    def _init_duckdb_views(self, con):
        """
        Initialize DuckDB views for the split Parquet files.
        """
        base_dir = self.infile

        # Gather files from multiple runs
        precursor_files = glob.glob(
            os.path.join(base_dir, "*.oswpq", "precursors_features.parquet")
        )
        transition_files = glob.glob(
            os.path.join(base_dir, "*.oswpq", "transition_features.parquet")
        )
        alignment_file = os.path.join(base_dir, "feature_alignment.parquet")
        print_alignment_file = (
            alignment_file if os.path.exists(alignment_file) else None
        )

        # If no multi-run structure, check single run input directory
        if not precursor_files:
            precursor_path = os.path.join(base_dir, "precursors_features.parquet")
            if os.path.exists(precursor_path):
                precursor_files = [precursor_path]
        if not transition_files:
            transition_path = os.path.join(base_dir, "transition_features.parquet")
            if os.path.exists(transition_path):
                transition_files = [transition_path]

        if self.context in ("score_learn", "ipf", "levels_context", "export"):
            print_parquet_tree(
                base_dir, precursor_files, transition_files, print_alignment_file, 5
            )

        if not precursor_files:
            raise click.ClickException("Error: No precursor Parquet files found.")

        # Create TEMP table of sampled precursor IDs (if needed)
        if self.subsample_ratio < 1.0:
            logger.info(
                f"Subsampling data for semi-supervised learning. Ratio: {self.subsample_ratio:.2f}"
            )
            con.execute(
                f"""
                CREATE TEMP TABLE sampled_precursor_ids AS
                SELECT DISTINCT PRECURSOR_ID
                FROM read_parquet({precursor_files})
                USING SAMPLE {self.subsample_ratio * 100}%
                """
            )
            n = con.execute("SELECT COUNT(*) FROM sampled_precursor_ids").fetchone()[0]
            logger.info(f"Sampled {n} precursor IDs")

        # Create view: precursors
        if self.subsample_ratio < 1.0:
            logger.trace("Creating 'precursors' view with sampled precursor IDs")
            con.execute(
                f"""
                CREATE VIEW precursors AS
                SELECT *
                FROM read_parquet({precursor_files})
                WHERE PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sampled_precursor_ids)
                """
            )
        else:
            logger.trace("Creating 'precursors' view with full input")
            con.execute(
                f"CREATE VIEW precursors AS SELECT * FROM read_parquet({precursor_files})"
            )

        # Create view: transition
        if transition_files and self.config.context in (
            "score_learn",
            "score_apply",
            "ipf",
            "export",
        ):
            if self.subsample_ratio < 1.0:
                logger.trace("Creating 'transition' view with sampled precursor IDs")
                con.execute(
                    f"""
                    CREATE VIEW transition AS
                    SELECT *
                    FROM read_parquet({transition_files})
                    WHERE PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sampled_precursor_ids)
                    """
                )
            else:
                logger.trace("Creating 'transition' view with full input")
                con.execute(
                    f"CREATE VIEW transition AS SELECT * FROM read_parquet({transition_files})"
                )

        # Create view: feature_alignment
        if os.path.exists(alignment_file) and self.config.context in (
            "score_learn",
            "score_apply",
            "ipf",
        ):
            if self.subsample_ratio < 1.0:
                logger.trace(
                    "Creating 'feature_alignment' view with sampled precursor IDs"
                )
                con.execute(
                    f"""
                    CREATE VIEW feature_alignment AS
                    SELECT *
                    FROM read_parquet('{alignment_file}')
                    WHERE PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sampled_precursor_ids)
                    """
                )
            else:
                logger.trace("Creating 'feature_alignment' view with full input")
                con.execute(
                    f"CREATE VIEW feature_alignment AS SELECT * FROM read_parquet('{alignment_file}')"
                )

    def _get_columns_by_prefix(self, parquet_file, prefix):
        """
        Returns columns that start with `prefix` from one of the parquet files.
        In multi-run mode, uses the first run's file as a representative.
        """
        if self._is_multi_run:
            candidate = glob.glob(
                os.path.join(self.config.infile, "*.oswpq", parquet_file)
            )
            if not candidate:
                raise click.ClickException(
                    f"Could not find '{parquet_file}' in any '.oswpq' subdirectory of '{self.config.infile}'."
                )
            path = candidate[0]
        else:
            path = os.path.join(self.config.infile, parquet_file)
            if not os.path.exists(path):
                raise click.ClickException(f"File '{path}' does not exist.")

        cols = get_parquet_column_names(path)
        logger.trace(f"Columns in {path}: {cols}")
        if cols is None:
            raise click.ClickException(f"Failed to read schema or columns from: {path}")

        return [c for c in cols if c.startswith(prefix)]


@dataclass
class BaseSplitParquetWriter(BaseParquetWriter):
    """
    Class for writing OpenSWATH results to a directory containing split Parquet files.

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

        if self.file_type not in ("parquet_split", "parquet_split_multi"):
            raise click.ClickException(
                f"SplitParquetWriter requires 'parquet_split' or 'parquet_split_multi' input, got '{self.file_type}' instead."
            )

        if self.context == "levels_context":
            if self.level not in ("peptide", "protein", "gene"):
                raise click.ClickException(
                    f"SplitParquetWriter levels_context only supports peptide, protein, or gene levels, got '{self.level}' instead."
                )

            self.context_level_id_map = {
                "peptide": "peptide_id",
                "protein": "protein_id",
                "gene": "gene_id",
            }

    def save_results(self, result, pi0=None):
        """
        Abstract method to be implemented by subclasses to save scoring results and statistical outputs.
        """
        raise NotImplementedError(
            "The save_results method must be implemented in subclasses of BaseSplitParquetWriter."
        )

    def merge_files(
        self,
        merge_transitions: bool = False,
    ):
        """
        Merges the precursors_features.parquet and transition_features.parquet files from all subdirectories into one single file each.
        """
        base_dir = self.infile

        if merge_transitions:
            out_dir = self.outfile
            os.makedirs(out_dir, exist_ok=True)
            output_file = os.path.join(out_dir, "precursors_features.parquet")
            output_file_transitions = os.path.join(
                out_dir, "transition_features.parquet"
            )
        else:
            output_file = self.outfile

        con = duckdb.connect()

        # Use glob to find all precursors and transition features files
        precursors_files = glob.glob(
            os.path.join(base_dir, "**", "precursors_features.parquet"), recursive=True
        )
        transition_files = glob.glob(
            os.path.join(base_dir, "**", "transition_features.parquet"), recursive=True
        )

        # Merge precursors features into one parquet file
        if precursors_files:
            logger.info(
                f"Merging {len(precursors_files)} precursors features files into {output_file}..."
            )
            precursors_query = f"""
                COPY (
                    SELECT * FROM read_parquet([{",".join([f"'{file}'" for file in precursors_files])}])
                ) TO '{output_file}' (FORMAT 'parquet')
            """
            logger.trace(f"Precursors merge query:\n{precursors_query}")
            con.execute(precursors_query)

        # Merge transition features into one parquet file (optional)
        if merge_transitions:
            logger.info(
                f"Merging {len(transition_files)} transition features files into {output_file_transitions}..."
            )
            transitions_query = f"""
                COPY (
                    SELECT * FROM read_parquet([{",".join([f"'{file}'" for file in transition_files])}])
                ) TO '{output_file_transitions}' (FORMAT 'parquet')
            """
            con.execute(transitions_query)

        con.close()
        logger.success("Merging complete.")
