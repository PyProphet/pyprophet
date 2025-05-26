import os
import pickle
from shutil import copyfile
import sqlite3
from typing import Literal
import duckdb
import pandas as pd
import click
from loguru import logger
from ..util import check_sqlite_table, check_duckdb_table
from .._base import BaseOSWReader, BaseOSWWriter
from ..._config import LevelContextIOConfig


class OSWReader(BaseOSWReader):
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

    def __init__(self, config: LevelContextIOConfig):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        self._create_indexes()
        try:
            con = duckdb.connect()
            con.execute("INSTALL sqlite_scanner;")
            con.execute("LOAD sqlite_scanner;")
            con.execute(f"ATTACH DATABASE '{self.infile}' AS osw (TYPE sqlite);")
            return self._read_using_duckdb(con)
        except ModuleNotFoundError as e:
            logger.warning(
                f"Warn: DuckDB sqlite_scanner failed, falling back to SQLite. Reason: {e}",
            )
            con = sqlite3.connect(self.infile)
            return self._read_using_sqlite(con)

    def _create_indexes(self):
        """
        Always use a temporary SQLite connection to create indexes directly on the .osw file,
        since DuckDB doesn't seem to currently support creating indexes on attached SQLite databases.
        """
        try:
            sqlite_con = sqlite3.connect(self.infile)

            if self.level == "peptide":
                index_statements = [
                    "CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);",
                ]
            elif self.level == "glycopeptide":
                index_statements = [
                    "CREATE INDEX IF NOT EXISTS idx_glycopeptide_glycopeptide_id ON GLYCOPEPTIDE (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_glycopeptide_mapping_glycopeptide_id ON PRECURSOR_GLYCOPEPTIDE_MAPPING (GLYCOPEPTIDE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_glycopeptide_mapping_precursor_id ON PRECURSOR_GLYCOPEPTIDE_MAPPING (PRECURSOR_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_score_ms2_part_peptide_feature_id ON SCORE_MS2_PART_PEPTIDE (FEATURE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_score_ms2_part_glycan_feature_id ON SCORE_MS2_PART_GLYCAN (FEATURE_ID);",
                ]
            elif self.level == "protein":
                index_statements = [
                    "CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_protein_id ON PEPTIDE_PROTEIN_MAPPING (PROTEIN_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_peptide_id ON PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);",
                ]
            elif self.level == "gene":
                index_statements = [
                    " CREATE INDEX IF NOT EXISTS idx_peptide_gene_mapping_gene_id ON PEPTIDE_GENE_MAPPING (GENE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_peptide_gene_mapping_peptide_id ON PEPTIDE_GENE_MAPPING (PEPTIDE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);",
                    "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);",
                    "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);",
                ]

            if index_statements:
                for stmt in index_statements:
                    try:
                        sqlite_con.execute(stmt)
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Warn: SQLite index creation failed: {e}")

            sqlite_con.commit()
            sqlite_con.close()

        except Exception as e:
            raise click.ClickException(
                f"Failed to create indexes via SQLite fallback: {e}"
            )

    def _read_using_duckdb(self, con):
        level = self.level
        if level == "peptide":
            return self._read_pyp_peptide_duckdb(con)
        elif level == "glycopeptide":
            return self._read_pyp_glycopeptide_duckdb(con)
        elif level == "protein":
            return self._read_pyp_protein_duckdb(con)
        elif level == "gene":
            return self._read_pyp_gene_duckdb(con)
        else:
            raise click.ClickException(f"Unsupported level: {level}")

    def _read_using_sqlite(self, con):
        level = self.level
        if level == "peptide":
            return self._read_pyp_peptide_sqlite(con)
        elif level == "glycopeptide":
            return self._read_pyp_glycopeptide_sqlite(con)
        elif level == "protein":
            return self._read_pyp_protein_sqlite(con)
        elif level == "gene":
            return self._read_pyp_gene_sqlite(con)
        else:
            raise click.ClickException(f"Unsupported level: {level}")

    # ----------------------------
    # DuckDB Queries
    # ----------------------------

    def _fetch_tables_duckdb(self, con):
        tables = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables"
        ).fetchdf()
        return tables

    def _read_pyp_peptide_duckdb(self, con):
        if not check_duckdb_table(con, "main", "SCORE_MS2"):
            raise click.ClickException(
                f"Apply scoring to MS2-level data before running peptide-level scoring.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )
        cfg = self.config  # LevelContextIOConfig instance
        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "PEPTIDE.ID"
        else:
            run_id = "RUN_ID"
            group_id = 'RUN_ID || "_" || PEPTIDE.ID'

        logger.info("Reading peptide-level data ...")

        query = f"""
                SELECT 
                    {run_id} AS RUN_ID,
                    {group_id} AS GROUP_ID,
                    PEPTIDE.ID AS PEPTIDE_ID,
                    PRECURSOR.DECOY,
                    SCORE_MS2.SCORE AS SCORE,
                    {cfg.context_fdr} AS CONTEXT
                FROM osw.PEPTIDE
                INNER JOIN osw.PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
                INNER JOIN osw.PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN osw.FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN osw.SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                GROUP BY GROUP_ID
                HAVING MAX(SCORE)
                ORDER BY SCORE DESC
            """

        df = con.execute(query).fetchdf()
        return df.rename(columns=str.lower)

    def _read_pyp_glycopeptide_duckdb(self, con):
        if (
            not check_duckdb_table(con, "main", "SCORE_MS2")
            or not check_duckdb_table(con, "main", "SCORE_MS2_PART_PEPTIDE")
            or not check_duckdb_table(con, "main", "SCORE_MS2_PART_GLYCAN")
        ):
            raise click.ClickException(
                f"Apply scoring to MS2-level data before running glycopeptide-level scoring.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )
        cfg = self.config
        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "GLYCOPEPTIDE.ID"
        else:
            run_id = "RUN_ID"
            group_id = 'RUN_ID || "_" || GLYCOPEPTIDE.ID'

        logger.info("Reading glycopeptide-level data ...")

        query = f"""
        SELECT 
            {run_id} AS RUN_ID,
            {group_id} AS GROUP_ID,
            GLYCOPEPTIDE.ID AS GLYCOPEPTIDE_ID,
            GLYCOPEPTIDE.DECOY_PEPTIDE,
            GLYCOPEPTIDE.DECOY_GLYCAN,
            SCORE_MS2.SCORE AS d_score_combined,
            SCORE_MS2_PART_PEPTIDE.SCORE AS d_score_peptide,
            SCORE_MS2_PART_GLYCAN.SCORE AS d_score_glycan,
            "{cfg.context_fdr}" AS CONTEXT
        FROM osw.GLYCOPEPTIDE
        INNER JOIN osw.PRECURSOR_GLYCOPEPTIDE_MAPPING ON GLYCOPEPTIDE.ID = PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID
        INNER JOIN osw.PRECURSOR ON PRECURSOR_GLYCOPEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN osw.FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
        INNER JOIN osw.SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        INNER JOIN osw.SCORE_MS2_PART_PEPTIDE ON FEATURE.ID = SCORE_MS2_PART_PEPTIDE.FEATURE_ID
        INNER JOIN osw.SCORE_MS2_PART_GLYCAN ON FEATURE.ID = SCORE_MS2_PART_GLYCAN.FEATURE_ID
        GROUP BY GROUP_ID
        HAVING MAX(d_score_combined)
        ORDER BY d_score_combined DESC
        """

        df = con.execute(query).fetchdf()
        return df.rename(columns=str.lower)

    def _read_pyp_protein_duckdb(self, con):
        if not check_duckdb_table(con, "main", "SCORE_MS2"):
            raise click.ClickException(
                f"Apply scoring to MS2-level data before running protein-level scoring.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )
        cfg = self.config

        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "PROTEIN.ID"
        else:
            run_id = "RUN_ID"
            group_id = 'RUN_ID || "_" || PROTEIN.ID'

        logger.info("Reading protein-level data ...")

        query = f"""
        SELECT {run_id} AS RUN_ID,
                {group_id} AS GROUP_ID,
                PROTEIN.ID AS PROTEIN_ID,
                PRECURSOR.DECOY AS DECOY,
                SCORE,
                "{cfg.context_fdr}" AS CONTEXT
        FROM osw.PROTEIN
        INNER JOIN
            (SELECT PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID AS PEPTIDE_ID,
                    PROTEIN_ID
            FROM
                (SELECT PEPTIDE_ID,
                        COUNT(*) AS NUM_PROTEINS
                FROM osw.PEPTIDE_PROTEIN_MAPPING
                GROUP BY PEPTIDE_ID) AS PROTEINS_PER_PEPTIDE
            INNER JOIN osw.PEPTIDE_PROTEIN_MAPPING ON PROTEINS_PER_PEPTIDE.PEPTIDE_ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
            WHERE NUM_PROTEINS == 1) AS PEPTIDE_PROTEIN_MAPPING ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
        INNER JOIN osw.PEPTIDE ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN osw.PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
        INNER JOIN osw.PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN osw.FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
        INNER JOIN osw.SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        GROUP BY GROUP_ID
        HAVING MAX(SCORE)
        ORDER BY SCORE DESC
        """

        df = con.execute(query).fetchdf()
        return df.rename(columns=str.lower)

    def _read_pyp_gene_duckdb(self, con):
        if not check_duckdb_table(con, "main", "SCORE_MS2"):
            raise click.ClickException(
                f"Apply scoring to MS2-level data before running gene-level scoring.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )
        cfg = self.config

        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "GENE.ID"
        else:
            run_id = "RUN_ID"
            group_id = 'RUN_ID || "_" || GENE.ID'

        logger.info("Reading gene-level data ...")

        query = f"""
        SELECT {run_id} AS RUN_ID,
                {group_id} AS GROUP_ID,
                GENE.ID AS GENE_ID,
                PRECURSOR.DECOY AS DECOY,
                SCORE,
                "{cfg.context_fdr}" AS CONTEXT
        FROM osw.GENE
        INNER JOIN
            (SELECT PEPTIDE_GENE_MAPPING.PEPTIDE_ID AS PEPTIDE_ID,
                    GENE_ID
            FROM
                (SELECT PEPTIDE_ID,
                        COUNT(*) AS NUM_GENES
                FROM osw.PEPTIDE_GENE_MAPPING
                GROUP BY PEPTIDE_ID) AS GENES_PER_PEPTIDE
            INNER JOIN osw.PEPTIDE_GENE_MAPPING ON GENES_PER_PEPTIDE.PEPTIDE_ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID
            WHERE NUM_GENES == 1) AS PEPTIDE_GENE_MAPPING ON GENE.ID = PEPTIDE_GENE_MAPPING.GENE_ID
        INNER JOIN osw.PEPTIDE ON PEPTIDE_GENE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN osw.PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
        INNER JOIN osw.PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN osw.FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
        INNER JOIN osw.SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        GROUP BY GROUP_ID
        HAVING MAX(SCORE)
        ORDER BY SCORE DESC
        """

        df = con.execute(query).fetchdf()
        return df.rename(columns=str.lower)

    # ----------------------------
    # SQLite fallback
    # ----------------------------

    def _read_pyp_peptide_sqlite(self, con):
        if not check_sqlite_table(con, "SCORE_MS2"):
            raise click.ClickException(
                "Apply scoring to MS2-level data before running peptide-level scoring."
            )

        cfg = self.config  # LevelContextIOConfig instance
        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "PEPTIDE.ID"
        else:
            run_id = "RUN_ID"
            group_id = 'RUN_ID || "_" || PEPTIDE.ID'

        logger.info("Reading peptide-level data ...")

        query = f"""
                SELECT
                    {run_id} AS RUN_ID,
                    {group_id} AS GROUP_ID,
                    PEPTIDE.ID AS PEPTIDE_ID,
                    PRECURSOR.DECOY,
                    SCORE_MS2.SCORE AS SCORE,
                    {cfg.context_fdr} AS CONTEXT
                FROM PEPTIDE
                INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
                INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
                INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                GROUP BY GROUP_ID
                HAVING MAX(SCORE)
                ORDER BY SCORE DESC
            """

        df = pd.read_sql_query(query, con)
        return df.rename(columns=str.lower)

    def _read_pyp_glycopeptide_sqlite(self, con):
        if (
            not check_sqlite_table(con, "SCORE_MS2")
            or not check_sqlite_table(con, "SCORE_MS2_PART_PEPTIDE")
            or not check_sqlite_table(con, "SCORE_MS2_PART_GLYCAN")
        ):
            raise click.ClickException(
                "Apply scoring to MS2-level data before running glycopeptide-level scoring."
            )
        cfg = self.config
        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "GLYCOPEPTIDE.ID"
        else:
            run_id = "RUN_ID"
            group_id = 'RUN_ID || "_" || GLYCOPEPTIDE.ID'

        logger.info("Reading glycopeptide-level data ...")

        query = f"""
        SELECT 
            {run_id} AS RUN_ID,
            {group_id} AS GROUP_ID,
            GLYCOPEPTIDE.ID AS GLYCOPEPTIDE_ID,
            GLYCOPEPTIDE.DECOY_PEPTIDE,
            GLYCOPEPTIDE.DECOY_GLYCAN,
            SCORE_MS2.SCORE AS d_score_combined,
            SCORE_MS2_PART_PEPTIDE.SCORE AS d_score_peptide,
            SCORE_MS2_PART_GLYCAN.SCORE AS d_score_glycan,
            "{cfg.context_fdr}" AS CONTEXT
        FROM GLYCOPEPTIDE
        INNER JOIN PRECURSOR_GLYCOPEPTIDE_MAPPING ON GLYCOPEPTIDE.ID = PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID
        INNER JOIN PRECURSOR ON PRECURSOR_GLYCOPEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
        INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        INNER JOIN SCORE_MS2_PART_PEPTIDE ON FEATURE.ID = SCORE_MS2_PART_PEPTIDE.FEATURE_ID
        INNER JOIN SCORE_MS2_PART_GLYCAN ON FEATURE.ID = SCORE_MS2_PART_GLYCAN.FEATURE_ID
        GROUP BY GROUP_ID
        HAVING MAX(d_score_combined)
        ORDER BY d_score_combined DESC
        """

        df = pd.read_sql_query(query, con)
        return df.rename(columns=str.lower)

    def _read_pyp_protein_sqlite(self, con):
        if not check_sqlite_table(con, "SCORE_MS2"):
            raise click.ClickException(
                "Apply scoring to MS2-level data before running protein-level scoring."
            )
        cfg = self.config
        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "PROTEIN.ID"
        else:
            run_id = "RUN_ID"
            group_id = 'RUN_ID || "_" || PROTEIN.ID'
        logger.info("Reading protein-level data ...")
        query = f"""
        SELECT {run_id} AS RUN_ID,
                {group_id} AS GROUP_ID,
                PROTEIN.ID AS PROTEIN_ID,
                PRECURSOR.DECOY AS DECOY,
                SCORE,
                "{cfg.context_fdr}" AS CONTEXT
        FROM PROTEIN
        INNER JOIN
            (SELECT PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID AS PEPTIDE_ID,
                    PROTEIN_ID
            FROM
                (SELECT PEPTIDE_ID,
                        COUNT(*) AS NUM_PROTEINS
                FROM PEPTIDE_PROTEIN_MAPPING
                GROUP BY PEPTIDE_ID) AS PROTEINS_PER_PEPTIDE    
            INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PROTEINS_PER_PEPTIDE.PEPTIDE_ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
            WHERE NUM_PROTEINS == 1) AS PEPTIDE_PROTEIN_MAPPING ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
        INNER JOIN PEPTIDE ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
        INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
        INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        GROUP BY GROUP_ID
        HAVING MAX(SCORE)
        ORDER BY SCORE DESC
        """
        df = pd.read_sql_query(query, con)
        return df.rename(columns=str.lower)

    def _read_pyp_gene_sqlite(self, con):
        if not check_sqlite_table(con, "SCORE_MS2"):
            raise click.ClickException(
                "Apply scoring to MS2-level data before running gene-level scoring."
            )
        cfg = self.config
        if cfg.context_fdr == "global":
            run_id = "NULL"
            group_id = "GENE.ID"
        else:
            run_id = "RUN_ID"
            group_id = 'RUN_ID || "_" || GENE.ID'
        logger.info("Reading gene-level data ...")
        query = f"""
        SELECT {run_id} AS RUN_ID,
                {group_id} AS GROUP_ID,
                GENE.ID AS GENE_ID,
                PRECURSOR.DECOY AS DECOY,
                SCORE,
                "{cfg.context_fdr}" AS CONTEXT
        FROM GENE
        INNER JOIN
            (SELECT PEPTIDE_GENE_MAPPING.PEPTIDE_ID AS PEPTIDE_ID,
                    GENE_ID
            FROM
                (SELECT PEPTIDE_ID,
                        COUNT(*) AS NUM_GENES
                FROM PEPTIDE_GENE_MAPPING
                GROUP BY PEPTIDE_ID) AS GENES_PER_PEPTIDE    
            INNER JOIN PEPTIDE_GENE_MAPPING ON GENES_PER_PEPTIDE.PEPTIDE_ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID
            WHERE NUM_GENES == 1) AS PEPTIDE_GENE_MAPPING ON GENE.ID = PEPTIDE_GENE_MAPPING.GENE_ID
        INNER JOIN PEPTIDE ON PEPTIDE_GENE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
        INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
        INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        GROUP BY GROUP_ID
        HAVING MAX(SCORE)
        ORDER BY SCORE DESC
        """
        df = pd.read_sql_query(query, con)
        return df.rename(columns=str.lower)


class OSWWriter(BaseOSWWriter):
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

    def __init__(self, config: LevelContextIOConfig):
        super().__init__(config)

        self.context_level_id_map = {
            "peptide": "peptide_id",
            "protein": "protein_id",
            "gene": "gene_id",
        }

    def save_results(self, result):
        """
        Save the results to the output file based on the module using this class.
        """
        context = self.config.context_fdr
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        con = sqlite3.connect(self.outfile)

        if self.level == "glycopeptide":
            c = con.cursor()
            c.execute(
                'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_GLYCOPEPTIDE"'
            )
            if c.fetchone()[0] == 1:
                c.execute(
                    'DELETE FROM SCORE_GLYCOPEPTIDE WHERE CONTEXT =="%s"' % context
                )
            c.fetchall()
            c.execute(
                'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_GLYCOPEPTIDE_PART_PEPTIDE"'
            )
            if c.fetchone()[0] == 1:
                c.execute(
                    'DELETE FROM SCORE_GLYCOPEPTIDE_PART_PEPTIDE WHERE CONTEXT =="%s"'
                    % context
                )
            c.fetchall()
            c.execute(
                'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_GLYCOPEPTIDE_PART_GLYCAN"'
            )
            if c.fetchone()[0] == 1:
                c.execute(
                    'DELETE FROM SCORE_GLYCOPEPTIDE_PART_GLYCAN WHERE CONTEXT =="%s"'
                    % context
                )
            c.fetchall()

            df = result[
                [
                    "context",
                    "run_id",
                    "glycopeptide_id",
                    "d_score_combined",
                    "q_value",
                    "pep",
                ]
            ]
            df.columns = [
                "CONTEXT",
                "RUN_ID",
                "GLYCOPEPTIDE_ID",
                "SCORE",
                "QVALUE",
                "PEP",
            ]
            table = "SCORE_GLYCOPEPTIDE"
            df.to_sql(
                table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists="append"
            )

            for part in ["peptide", "glycan"]:
                df = result[
                    [
                        "context",
                        "run_id",
                        "glycopeptide_id",
                        "d_score_" + part,
                        "pep_" + part,
                    ]
                ]
                df.columns = ["CONTEXT", "RUN_ID", "GLYCOPEPTIDE_ID", "SCORE", "PEP"]
                table = "SCORE_GLYCOPEPTIDE_PART_" + part.upper()
                df.to_sql(
                    table,
                    con,
                    index=False,
                    dtype={"RUN_ID": "INTEGER"},
                    if_exists="append",
                )

        else:
            c = con.cursor()
            c.execute(
                f'''SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_{self.level.upper()}"'''
            )
            if c.fetchone()[0] == 1:
                c.execute(
                    f'''DELETE FROM SCORE_{self.level.upper()} WHERE CONTEXT =="{context}"'''
                )
            c.fetchall()

            context_level_id = self.context_level_id_map.get(self.level)

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
            result.columns = [
                "CONTEXT",
                "RUN_ID",
                context_level_id.upper(),
                "SCORE",
                "PVALUE",
                "QVALUE",
                "PEP",
            ]

            result.to_sql(
                f"SCORE_{self.level.upper()}",
                con,
                index=False,
                dtype={"RUN_ID": "INTEGER"},
                if_exists="replace",
            )
            con.close()
