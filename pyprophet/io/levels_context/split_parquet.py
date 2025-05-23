import os
import glob
from shutil import copyfile
from typing import Literal
import pandas as pd
import pyarrow as pa
import duckdb
import click
from loguru import logger

from .._base import BaseReader, BaseWriter
from ..._config import LevelContextIOConfig
from ..util import (
    setup_logger,
    print_parquet_tree,
    get_parquet_column_names,
)

setup_logger()


class SplitParquetReader(BaseReader):
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

    def __init__(self, config: LevelContextIOConfig):
        super().__init__(config)
        self.config = config

        if config.file_type not in ("parquet_split", "parquet_split_multi"):
            raise click.ClickException(
                f"SplitParquetReader requires 'parquet_split' or 'parquet_split_multi' input, got '{config.file_type}' instead."
            )

        # Flag to indicate whether the input is a multi-run directory
        self._is_multi_run = config.file_type == "parquet_split_multi"

    def read(self) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            self._init_duckdb_views(con)

            level = self.level
            if level == "peptide":
                return self._read_pyp_peptide(con)
            elif level == "glycopeptide":
                raise click.ClickException(
                    "Glycopeptide-level inference is not supported for split parquet files."
                )
            elif level == "protein":
                return self._read_pyp_protein(con)
            elif level == "gene":
                return self._read_pyp_gene(con)
            else:
                raise click.ClickException(f"Unsupported level: {level}")
        finally:
            con.close()

    def _init_duckdb_views(self, con):
        base_dir = self.infile

        # Gather files from multiple runs
        precursor_files = glob.glob(
            os.path.join(base_dir, "*.oswpq", "precursors_features.parquet")
        )

        # If no multi-run structure, check single run input directory
        if not precursor_files:
            precursor_path = os.path.join(base_dir, "precursors_features.parquet")
            if os.path.exists(precursor_path):
                precursor_files = [precursor_path]

        # print_parquet_tree(base_dir, precursor_files, transition_files, alignment_file)

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
            logger.debug("Creating 'precursors' view with sampled precursor IDs")
            con.execute(
                f"""
                CREATE VIEW precursors AS
                SELECT *
                FROM read_parquet({precursor_files})
                WHERE PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sampled_precursor_ids)
                """
            )
        else:
            logger.debug("Creating 'precursors' view with full input")
            con.execute(
                f"CREATE VIEW precursors AS SELECT * FROM read_parquet({precursor_files})"
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
        if cols is None:
            raise click.ClickException(f"Failed to read schema or columns from: {path}")

        return [c for c in cols if c.startswith(prefix)]

    def _read_pyp_peptide(self, con) -> pd.DataFrame:
        cfg = self.config  # LevelContextIOConfig instance

        if cfg.file_type == "parquet_split_multi":
            precursor_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "precursors_features.parquet")
            )
        else:
            precursor_files = [os.path.join(self.infile, "precursors_features.parquet")]

        all_precursor_cols = get_parquet_column_names(precursor_files[0])

        if not any(c.startswith("SCORE_MS2_") for c in all_precursor_cols):
            raise click.ClickException(
                "Apply scoring to MS2-level data before running peptide-level scoring."
            )

        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "PEPTIDE_ID"
        else:
            run_id = "RUN_ID"
            group_id = "RUN_ID || '_' || PEPTIDE_ID"

        logger.info("Reading peptide-level data ...")
        query = f"""
            SELECT
                {run_id} AS RUN_ID,
                {group_id} AS GROUP_ID,
                PEPTIDE_ID AS PEPTIDE_ID,
                PRECURSOR_DECOY AS DECOY,
                SCORE_MS2_SCORE AS SCORE,
                '{cfg.context_fdr}' AS CONTEXT
            FROM precursors p
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {group_id} ORDER BY SCORE_MS2_SCORE DESC) = 1
        """

        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]
        return df

    def _read_pyp_protein(self, con) -> pd.DataFrame:
        cfg = self.config
        if cfg.file_type == "parquet_split_multi":
            precursor_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "precursors_features.parquet")
            )
        else:
            precursor_files = [os.path.join(self.infile, "precursors_features.parquet")]
        all_precursor_cols = get_parquet_column_names(precursor_files[0])
        if not any(c.startswith("SCORE_MS2_") for c in all_precursor_cols):
            raise click.ClickException(
                "Apply scoring to MS2-level data before running protein-level scoring."
            )

        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "PROTEIN_ID"
        else:
            run_id = "RUN_ID"
            group_id = "RUN_ID || '_' || PROTEIN_ID"

        logger.info("Reading protein-level data ...")
        query = f"""
            with one_peptide_proteins AS (
                SELECT PEPTIDE_ID
                FROM precursors
                WHERE PROTEIN_ID IS NOT NULL
                GROUP BY PEPTIDE_ID
                HAVING COUNT(DISTINCT PROTEIN_ID) = 1
            )
            SELECT
                {run_id} AS RUN_ID,
                {group_id} AS GROUP_ID,
                PROTEIN_ID AS PROTEIN_ID,
                PRECURSOR_DECOY AS DECOY,
                SCORE_MS2_SCORE AS SCORE,
                {cfg.context_fdr} AS CONTEXT
            FROM precursors p
            JOIN one_peptide_proteins opp ON p.PEPTIDE_ID = opp.PEPTIDE_ID
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {group_id} ORDER BY SCORE_MS2_SCORE DESC) = 1
        """
        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]
        return df

    def _read_pyp_gene(self, con) -> pd.DataFrame:
        cfg = self.config
        if cfg.file_type == "parquet_split_multi":
            precursor_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "precursors_features.parquet")
            )
        else:
            precursor_files = [os.path.join(self.infile, "precursors_features.parquet")]
        all_precursor_cols = get_parquet_column_names(precursor_files[0])
        if not any(c.startswith("SCORE_MS2_") for c in all_precursor_cols):
            raise click.ClickException(
                "Apply scoring to MS2-level data before running gene-level scoring."
            )

        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "GENE_ID"
        else:
            run_id = "RUN_ID"
            group_id = "RUN_ID || '_' || GENE_ID"

        logger.info("Reading gene-level data ...")
        query = f"""
            WITH one_gene_peptides AS (
                SELECT PEPTIDE_ID
                FROM precursors
                WHERE GENE_ID IS NOT NULL
                GROUP BY PEPTIDE_ID
                HAVING COUNT(DISTINCT GENE_ID) = 1
            )
            SELECT
                {run_id} AS RUN_ID,
                {group_id} AS GROUP_ID,
                GENE_ID AS GENE_ID,
                PRECURSOR_DECOY AS DECOY,
                SCORE_MS2_SCORE AS SCORE,
                {cfg.context_fdr} AS CONTEXT
            FROM precursors p
            JOIN one_gene_peptides ogp ON p.PEPTIDE_ID = ogp.PEPTIDE_ID
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {group_id} ORDER BY SCORE_MS2_SCORE DESC) = 1
        """
        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]
        return df


class SplitParquetWriter(BaseWriter):
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

    def __init__(self, config: LevelContextIOConfig):
        super().__init__(config)

        if self.file_type not in ("parquet_split", "parquet_split_multi"):
            raise click.ClickException(
                f"SplitParquetWriter requires 'parquet_split' or 'parquet_split_multi' input, got '{self.file_type}' instead."
            )

        if self.level not in ("peptide", "protein", "gene"):
            raise click.ClickException(
                f"SplitParquetWriter levels_context only supports peptide, protein, or gene levels, got '{self.level}' instead."
            )

        self.context_level_id_map = {
            "peptide": "peptide_id",
            "protein": "protein_id",
            "gene": "gene_id",
        }

    def save_results(self, result):

        context = self.config.context_fdr
        context_level_id = self.context_level_id_map.get(self.level)
        col_prefix = f"SCORE_{self.level.upper()}_{context.upper().replace('-', '_')}"

        result = result[
            [
                "context",
                "run_id",
                context_level_id,
                "score",
                "p_value",
                "q_value",
                "pep",
            ]
        ]
        # drop context column
        result = result.drop(columns=["context"])

        result.columns = [col.upper() for col in result.columns]

        result = result.rename(
            columns=lambda x: (
                f"{col_prefix}_{x}"
                if x not in ("RUN_ID", context_level_id.upper())
                else x
            )
        )

        score_cols = [
            col
            for col in result.columns
            if col.startswith(col_prefix) and col != "RUN_ID"
        ]

        # Determine output files to modify
        if self.file_type == "parquet_split_multi":
            run_dirs = [
                os.path.join(self.outfile, d)
                for d in os.listdir(self.outfile)
                if d.endswith(".oswpq") and os.path.isdir(os.path.join(self.outfile, d))
            ]
        else:
            run_dirs = [self.outfile]

        for run_dir in run_dirs:
            file_path = os.path.join(run_dir, "precursors_features.parquet")

            if not os.path.exists(file_path):
                logger.warning(f"File not found, skipping: {file_path}")
                continue

            # Identify columns to keep from original parquet file
            existing_cols = get_parquet_column_names(file_path)
            exitsting_score_cols = [
                col for col in existing_cols if col.startswith(col_prefix)
            ]
            if exitsting_score_cols:
                logger.warning(
                    f"Warn: There are existing {col_prefix}_ columns, these will be dropped."
                )
            existing_cols = [
                col for col in existing_cols if not col.startswith(col_prefix)
            ]
            select_old = ", ".join([f"p.{col}" for col in existing_cols])
            new_score_sql = ", ".join([f"s.{col}" for col in score_cols])

            con = duckdb.connect()
            con.register("scores", pa.Table.from_pandas(result))

            if context == "global":
                con.execute(
                    f"""
                    COPY (
                        SELECT {select_old}, {new_score_sql}
                        FROM read_parquet('{file_path}') p
                        LEFT JOIN scores s
                        ON p.{context_level_id.upper()} = s.{context_level_id.upper()}
                    ) TO '{file_path}'
                    (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11)
                    """
                )
            else:
                # Read RUN_IDs from current file
                try:
                    con_tmp = duckdb.connect()
                    run_ids = con.execute(
                        f"SELECT DISTINCT RUN_ID FROM read_parquet('{file_path}')"
                    ).fetchall()
                    con_tmp.close()
                except Exception as e:
                    logger.error(f"Error reading RUN_IDs from {file_path}: {e}")
                    continue

                run_ids = set(f[0] for f in run_ids)
                subset = result[result["RUN_ID"].isin(run_ids)]

                if subset.empty:
                    logger.warning(
                        f"No matching RUN_IDs found for {run_dir}, skipping."
                    )
                    continue

                con.execute(
                    f"""
                    COPY (
                        SELECT {select_old}, {new_score_sql}
                        FROM read_parquet('{file_path}') p
                        LEFT JOIN scores s
                        ON p.RUN_ID = s.RUN_ID AND p.{context_level_id.upper()} = s.{context_level_id.upper()}
                    ) TO '{file_path}'
                    (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11)
                    """
                )
            con.close()

            logger.success(f"Updated: {file_path}")
