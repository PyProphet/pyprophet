import os
from typing import Literal
from shutil import copyfile
import pandas as pd
import pyarrow as pa
import duckdb
import click
from loguru import logger
from ..util import get_parquet_column_names
from .._base import BaseParquetReader, BaseParquetWriter
from ..._config import LevelContextIOConfig


class ParquetReader(BaseParquetReader):
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

    def __init__(self, config: LevelContextIOConfig):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            self._init_duckdb_views(con)

            level = self.config.level
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

    def _read_pyp_peptide(self, con) -> pd.DataFrame:
        cfg = self.config  # LevelContextIOConfig instance

        all_cols = get_parquet_column_names(self.infile)
        if not any(c.startswith("SCORE_MS2_") for c in all_cols):
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
            FROM data p
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {group_id} ORDER BY SCORE_MS2_SCORE DESC) = 1
        """

        df = con.execute(query).df()
        return df.rename(columns=str.lower)

    def _read_pyp_protein(self, con) -> pd.DataFrame:
        cfg = self.config  # LevelContextIOConfig instance

        all_cols = get_parquet_column_names(self.infile)
        if not any(c.startswith("SCORE_MS2_") for c in all_cols):
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
                FROM data
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
                '{cfg.context_fdr}' AS CONTEXT
            FROM data p
            JOIN one_peptide_proteins opp ON p.PEPTIDE_ID = opp.PEPTIDE_ID
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {group_id} ORDER BY SCORE_MS2_SCORE DESC) = 1
        """
        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]
        return df

    def _read_pyp_gene(self, con) -> pd.DataFrame:
        cfg = self.config  # LevelContextIOConfig instance

        all_cols = get_parquet_column_names(self.infile)
        if not any(c.startswith("SCORE_MS2_") for c in all_cols):
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
                FROM data
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
                '{cfg.context_fdr}' AS CONTEXT
            FROM data p
            JOIN one_gene_peptides ogp ON p.PEPTIDE_ID = ogp.PEPTIDE_ID
            QUALIFY ROW_NUMBER() OVER (PARTITION BY {group_id} ORDER BY SCORE_MS2_SCORE DESC) = 1
        """
        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]
        return df


class ParquetWriter(BaseParquetWriter):
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

    def __init__(self, config: LevelContextIOConfig):
        super().__init__(config)

    def save_results(self, result):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        file_path = self.outfile

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

        existing_cols = get_parquet_column_names(file_path)
        exitsting_score_cols = [
            col for col in existing_cols if col.startswith(col_prefix)
        ]
        if exitsting_score_cols:
            logger.warning(
                f"Warn: There are existing {col_prefix}_ columns, these will be dropped."
            )
        existing_cols = [col for col in existing_cols if not col.startswith(col_prefix)]
        select_old = ", ".join([f"p.{col}" for col in existing_cols])
        new_score_sql = ", ".join([f"s.{col}" for col in score_cols])

        con = duckdb.connect()
        con.register("scores", pa.Table.from_pandas(result))

        if context == "global":
            # Validate input row entry count and joined entry count remain the same
            self._validate_row_count_after_join(
                con,
                file_path,
                f"p.{context_level_id.upper()}",
                f"p.{context_level_id.upper()} = s.{context_level_id.upper()}",
                "p",
            )

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
            # Validate input row entry count and joined entry count remain the same
            self._validate_row_count_after_join(
                con,
                file_path,
                f"p.RUN_ID, p.{context_level_id.upper()}",
                f"p.RUN_ID = s.RUN_ID AND p.{context_level_id.upper()} = s.{context_level_id.upper()}",
                "p",
            )
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
        logger.debug(
            f"After appendings scores, {file_path} has {self._get_parquet_row_count(con, file_path)} entries"
        )
        con.close()

        logger.success(f"Updated: {file_path}")
