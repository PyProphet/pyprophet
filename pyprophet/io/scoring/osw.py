import os
import pickle
from shutil import copyfile
import sqlite3
import zlib
import duckdb
import pandas as pd
import click
from loguru import logger
from ..util import check_sqlite_table, check_duckdb_table, get_table_columns
from .._base import BaseOSWReader, BaseOSWWriter
from ..._config import RunnerIOConfig


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

    def __init__(self, config: RunnerIOConfig):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        """
        Reads the data for scoring from the specified file using DuckDB if available,
        falling back to SQLite if DuckDB is not available.

        Returns:
        pd.DataFrame: The data read from the file.
        """
        self._create_indexes()
        if getattr(self.config, "run_id_filter", None) is not None:
            logger.info(
                "Using SQLite read path for run-scoped OSW access."
            )
            con = sqlite3.connect(self.infile)
            return self._read_using_sqlite(con)
        try:
            con = duckdb.connect()
            con.execute("INSTALL sqlite_scanner;")
            con.execute("LOAD sqlite_scanner;")
            con.execute(f"ATTACH DATABASE '{self.infile}' AS osw (TYPE sqlite);")
            self._init_duckdb_views(con)
            return self._read_using_duckdb(con)
        except ModuleNotFoundError as e:
            logger.warning(
                f"Warn: DuckDB sqlite_scanner failed, falling back to SQLite. Reason: {e}"
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

            index_statements = [
                "CREATE INDEX IF NOT EXISTS idx_precursor_id ON PRECURSOR (ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_run_id ON FEATURE (RUN_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_run_id_feature_id ON FEATURE (RUN_ID, ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id_rank_pep ON SCORE_MS2 (FEATURE_ID, RANK, PEP);",
                "CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id_transition_id ON FEATURE_TRANSITION (FEATURE_ID, TRANSITION_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);",
                "CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);",
            ]

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
        """
        Read features from SQLite using DuckDB based on the specified level.

        Parameters:
        - con: Connection to the DuckDB database.

        Returns:
        - Features based on the specified level.

        Raises:
        - click.ClickException: If the specified level is unsupported.
        """
        level = self.level
        if level in ("ms2", "ms1ms2"):
            return self._fetch_ms2_features_duckdb(con)
        elif level == "ms1":
            return self._fetch_ms1_features_duckdb(con)
        elif level == "transition":
            return self._fetch_transition_features_duckdb(con)
        elif level == "alignment":
            return self._fetch_alignment_features_duckdb(con)
        else:
            raise click.ClickException(f"Unsupported level: {level}")

    def _read_using_sqlite(self, con):
        """
        Read features from SQLite database based on the specified level.

        Parameters:
        - con: SQLite connection object

        Returns:
        - Features based on the specified level

        Raises:
        - click.ClickException: If the specified level is unsupported
        """
        level = self.level
        if level in ("ms2", "ms1ms2"):
            return self._fetch_ms2_features_sqlite(con)
        elif level == "ms1":
            return self._fetch_ms1_features_sqlite(con)
        elif level == "transition":
            return self._fetch_transition_features_sqlite(con)
        elif level == "alignment":
            return self._fetch_alignment_features_sqlite(con)
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

    def _get_precursor_filter_clause(self):
        """
        Return a WHERE/AND clause fragment for filtering by sampled precursor IDs when subsampling is enabled.
        Returns empty string if no subsampling, otherwise returns a clause like:
        " AND f.PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sampled_precursor_ids)"
        """
        if self.subsample_ratio < 1.0:
            return " AND f.PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sampled_precursor_ids)"
        return ""

    def _get_run_filter_clause(self, alias="f"):
        run_filter = getattr(self.config, "run_id_filter", None)
        if run_filter is None:
            return ""

        if isinstance(run_filter, (list, tuple, set)):
            try:
                run_ids = tuple(int(run_id) for run_id in run_filter)
            except (TypeError, ValueError) as exc:
                raise click.ClickException(
                    f"Invalid run_id_filter value: {run_filter}"
                ) from exc
            if not run_ids:
                return ""
            if len(run_ids) == 1:
                return f" AND {alias}.RUN_ID = {run_ids[0]}"
            return (
                f" AND {alias}.RUN_ID IN ("
                + ",".join(str(run_id) for run_id in run_ids)
                + ")"
            )

        try:
            run_id = int(run_filter)
        except (TypeError, ValueError) as exc:
            raise click.ClickException(
                f"Invalid run_id_filter value: {run_filter}"
            ) from exc

        return f" AND {alias}.RUN_ID = {run_id}"

    def _fetch_ms2_features_duckdb(self, con):
        if not check_duckdb_table(con, "main", "FEATURE_MS2"):
            raise click.ClickException(
                f"MS2-level feature table not present in file.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )

        filter_clause = self._get_precursor_filter_clause()
        run_filter_clause = self._get_run_filter_clause("f")

        if self.glyco:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW ms2_table AS
                SELECT
                    fm.*,
                    f.RUN_ID,
                    f.PRECURSOR_ID,
                    f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE,
                    p.DECOY,
                    g.DECOY_PEPTIDE,
                    g.DECOY_GLYCAN,
                    COALESCE(ts.TRANSITION_COUNT, 0) AS TRANSITION_COUNT,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM osw.FEATURE_MS2 fm
                INNER JOIN osw.FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN osw.PRECURSOR p ON f.PRECURSOR_ID = p.ID
                INNER JOIN (
                    SELECT
                        pgm.PRECURSOR_ID,
                        gp.DECOY_PEPTIDE,
                        gp.DECOY_GLYCAN
                    FROM osw.PRECURSOR_GLYCOPEPTIDE_MAPPING pgm
                    INNER JOIN osw.GLYCOPEPTIDE gp ON pgm.GLYCOPEPTIDE_ID = gp.ID
                ) g ON f.PRECURSOR_ID = g.PRECURSOR_ID
                LEFT JOIN (
                    SELECT
                        tpm.PRECURSOR_ID,
                        COUNT(*) AS TRANSITION_COUNT
                    FROM osw.TRANSITION_PRECURSOR_MAPPING tpm
                    INNER JOIN osw.TRANSITION t ON tpm.TRANSITION_ID = t.ID
                    WHERE t.DETECTING = 1
                    GROUP BY tpm.PRECURSOR_ID
                ) ts ON f.PRECURSOR_ID = ts.PRECURSOR_ID
                WHERE 1=1{filter_clause}{run_filter_clause}
                """
            )
        else:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW ms2_table AS
                SELECT
                    fm.*,
                    f.RUN_ID,
                    f.PRECURSOR_ID,
                    f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE,
                    p.DECOY,
                    COALESCE(ts.TRANSITION_COUNT, 0) AS TRANSITION_COUNT,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM osw.FEATURE_MS2 fm
                INNER JOIN osw.FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN osw.PRECURSOR p ON f.PRECURSOR_ID = p.ID
                LEFT JOIN (
                    SELECT
                        tpm.PRECURSOR_ID,
                        COUNT(*) AS TRANSITION_COUNT
                    FROM osw.TRANSITION_PRECURSOR_MAPPING tpm
                    INNER JOIN osw.TRANSITION t ON tpm.TRANSITION_ID = t.ID
                    WHERE t.DETECTING = 1
                    GROUP BY tpm.PRECURSOR_ID
                ) ts ON f.PRECURSOR_ID = ts.PRECURSOR_ID
                WHERE 1=1{filter_clause}{run_filter_clause}
                """
            )

        df = con.execute(
            "SELECT * FROM ms2_table ORDER BY RUN_ID, PRECURSOR_ID, EXP_RT"
        ).fetchdf()

        if self.level == "ms1ms2":
            ms1_df = con.execute(
                f"""
                SELECT fm1.*
                FROM osw.FEATURE_MS1 fm1
                INNER JOIN osw.FEATURE f ON fm1.FEATURE_ID = f.ID
                WHERE 1=1{filter_clause}{run_filter_clause}
                """
            ).fetchdf()
            ms1_scores = [c for c in ms1_df.columns if c.startswith("VAR_")]
            ms1_df = ms1_df[["FEATURE_ID"] + ms1_scores]
            ms1_df.columns = ["FEATURE_ID"] + [
                "VAR_MS1_" + s.split("VAR_")[1] for s in ms1_scores
            ]
            df = pd.merge(df, ms1_df, how="left", on="FEATURE_ID")

        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _fetch_ms1_features_duckdb(self, con):
        if not check_duckdb_table(con, "main", "FEATURE_MS1"):
            raise click.ClickException(
                f"MS1-level feature table not present in file.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )

        rc = self.config.runner
        glyco = rc.glyco
        ipf_max_rank = rc.ipf_max_peakgroup_rank
        filter_clause = self._get_precursor_filter_clause()
        run_filter_clause = self._get_run_filter_clause("f")

        if not glyco:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW ms1_table AS
                SELECT fm.*, f.RUN_ID, f.PRECURSOR_ID, f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE, p.DECOY,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM osw.FEATURE_MS1 fm
                INNER JOIN osw.FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN osw.PRECURSOR p ON f.PRECURSOR_ID = p.ID
                WHERE 1=1{filter_clause}{run_filter_clause}
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
                """
            )
        else:
            if not check_duckdb_table(con, "main", "SCORE_MS2"):
                raise click.ClickException(
                    f"""MS1-level scoring for glycoform inference requires prior MS2 or MS1MS2-level scoring.\n
                    Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first.level\nTable Info:\n{self._fetch_tables_duckdb(con)}"""
                )

            con.execute(
                f"""
                CREATE OR REPLACE VIEW ms1_table AS
                SELECT g.DECOY_PEPTIDE, g.DECOY_GLYCAN,
                    fm.*, f.*, p.*,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM osw.FEATURE_MS1 fm
                INNER JOIN osw.FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN osw.SCORE_MS2 s ON f.ID = s.FEATURE_ID
                INNER JOIN osw.PRECURSOR p ON f.PRECURSOR_ID = p.ID
                INNER JOIN (
                    SELECT pgm.PRECURSOR_ID,
                        gp.DECOY_PEPTIDE,
                        gp.DECOY_GLYCAN
                    FROM osw.PRECURSOR_GLYCOPEPTIDE_MAPPING pgm
                    INNER JOIN osw.GLYCOPEPTIDE gp ON pgm.GLYCOPEPTIDE_ID = gp.ID
                ) g ON f.PRECURSOR_ID = g.PRECURSOR_ID
                WHERE s.RANK <= {ipf_max_rank}{filter_clause}{run_filter_clause}
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
                """
            )

        df = con.execute("SELECT * FROM ms1_table").fetchdf()

        return self._finalize_feature_table(df, rc.ss_main_score)

    def _fetch_transition_features_duckdb(self, con):
        if not check_duckdb_table(con, "main", "SCORE_MS2"):
            raise click.ClickException(
                f"""Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring.\n
                Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' first.\nTable Info:\n{self._fetch_tables_duckdb(con)}"""
            )

        if not check_duckdb_table(con, "main", "FEATURE_TRANSITION"):
            raise click.ClickException(
                f"Transition-level feature table not present in file.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )

        rc = self.config.runner
        filter_clause = self._get_precursor_filter_clause()
        run_filter_clause = self._get_run_filter_clause("f")
        include_mapping_cardinality = rc.transition_score_use_mapping_cardinality
        include_unique_mapping = rc.transition_score_use_unique_mapping
        include_phospho_loss = rc.transition_score_use_phospho_loss
        need_training_unique = rc.transition_training_require_unique_mapping
        need_training_phospho_loss = rc.transition_training_require_phospho_loss
        need_mapping_counts = (
            include_mapping_cardinality
            or include_unique_mapping
            or need_training_unique
        )
        transition_cols = set(get_table_columns(self.infile, "TRANSITION"))
        extra_select_parts = []
        if include_mapping_cardinality:
            extra_select_parts.append(
                "COALESCE(tmc.N_MAPPED_PEPTIDES, 0) AS VAR_MAPPING_CARDINALITY"
            )
        if include_unique_mapping:
            extra_select_parts.append(
                """CASE
                    WHEN COALESCE(tmc.N_MAPPED_PEPTIDES, 0) = 1 THEN 1.0
                    ELSE 0.0
                END AS VAR_IS_UNIQUE_MAPPING"""
            )
        if need_training_unique:
            extra_select_parts.append(
                """CASE
                    WHEN COALESCE(tmc.N_MAPPED_PEPTIDES, 0) = 1 THEN 1.0
                    ELSE 0.0
                END AS meta_is_unique_mapping"""
            )
        if include_phospho_loss or need_training_phospho_loss:
            if "ANNOTATION" in transition_cols:
                extra_select_parts.append("tr.ANNOTATION AS TRANSITION_ANNOTATION")
            else:
                extra_select_parts.append("NULL AS TRANSITION_ANNOTATION")
        extra_select_sql = ""
        if extra_select_parts:
            extra_select_sql = ",\n                " + ",\n                ".join(extra_select_parts)
        mapping_join_sql = ""
        if need_mapping_counts:
            mapping_join_sql = """
            LEFT JOIN (
                SELECT
                    TRANSITION_ID,
                    COUNT(DISTINCT PEPTIDE_ID) AS N_MAPPED_PEPTIDES
                FROM osw.TRANSITION_PEPTIDE_MAPPING
                GROUP BY TRANSITION_ID
            ) tmc ON ft.TRANSITION_ID = tmc.TRANSITION_ID
            """
        con.execute(
            f"""
            CREATE OR REPLACE VIEW transition_table AS
            SELECT 
                ft.*{extra_select_sql},
                tr.DECOY AS DECOY,
                f.RUN_ID,
                f.PRECURSOR_ID,
                f.EXP_RT,
                p.CHARGE AS PRECURSOR_CHARGE,
                tr.CHARGE AS PRODUCT_CHARGE,
                f.RUN_ID || '_' || ft.FEATURE_ID || '_' || f.PRECURSOR_ID || '_' || ft.TRANSITION_ID AS GROUP_ID
            FROM osw.FEATURE_TRANSITION ft
            INNER JOIN osw.FEATURE f ON ft.FEATURE_ID = f.ID
            INNER JOIN osw.SCORE_MS2 s ON f.ID = s.FEATURE_ID
            INNER JOIN osw.PRECURSOR p ON f.PRECURSOR_ID = p.ID
            INNER JOIN osw.TRANSITION tr ON ft.TRANSITION_ID = tr.ID
            {mapping_join_sql}
            WHERE s.RANK <= {rc.ipf_max_peakgroup_rank}
            AND s.PEP <= {rc.ipf_max_peakgroup_pep}
            AND ft.VAR_ISOTOPE_OVERLAP_SCORE <= {rc.ipf_max_transition_isotope_overlap}
            AND ft.VAR_LOG_SN_SCORE > {rc.ipf_min_transition_sn}
            AND p.DECOY = 0{filter_clause}{run_filter_clause}
            """
        )
        df = con.execute(
            """
            SELECT * 
            FROM transition_table 
            ORDER BY RUN_ID, FEATURE_ID, PRECURSOR_ID, EXP_RT, TRANSITION_ID
            """
        ).fetchdf()
        if include_phospho_loss or need_training_phospho_loss:
            transition_annotation = df["TRANSITION_ANNOTATION"].astype("string")
            phospho_loss = (
                transition_annotation
                .fillna("")
                .str.contains("-H3O4P1", regex=False)
                .astype(float)
            )
            if include_phospho_loss:
                df["VAR_HAS_PHOSPHO_LOSS"] = phospho_loss
            if need_training_phospho_loss:
                df["meta_has_phospho_loss"] = phospho_loss
            df = df.drop(columns=["TRANSITION_ANNOTATION"])

        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _fetch_alignment_features_duckdb(self, con):
        if not check_duckdb_table(con, "main", "FEATURE_MS2_ALIGNMENT"):
            raise click.ClickException(
                f"MS2-level feature alignment table not present in file.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )
        
        filter_clause = self._get_precursor_filter_clause()
        run_filter_clause = self._get_run_filter_clause("fa")
        con.execute(
            f"""
            CREATE OR REPLACE VIEW alignment_table AS
            SELECT
                fa.ALIGNMENT_ID AS ALIGNMENT_ID, fa.RUN_ID,
                fa.PRECURSOR_ID, fa.ALIGNED_FEATURE_ID AS FEATURE_ID,
                fa.ALIGNED_RT, fa.LABEL AS DECOY,
                fa.XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
                fa.XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE,
                fa.MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE,
                fa.XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,
                fa.XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE_TO_ALL,
                fa.MI_TO_ALL AS VAR_MI_TO_ALL,
                fa.RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION,
                fa.PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO,
                fa.ALIGNED_FEATURE_ID || '_' || fa.PRECURSOR_ID AS GROUP_ID
            FROM osw.FEATURE_MS2_ALIGNMENT fa
            WHERE 1=1{filter_clause}{run_filter_clause}
            ORDER BY fa.RUN_ID, fa.PRECURSOR_ID, fa.REFERENCE_RT
        """
        )
        df = con.execute("SELECT * FROM alignment_table").fetchdf()
        df["DECOY"] = df["DECOY"].map({1: 0, -1: 1})
        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    # ----------------------------
    # SQLite fallback
    # ----------------------------

    def _fetch_ms2_features_sqlite(self, con):
        if not check_sqlite_table(con, "FEATURE_MS2"):
            raise click.ClickException("MS2-level feature table not present in file.")

        run_filter_clause = self._get_run_filter_clause("f")

        if not self.glyco:
            query = f"""
                SELECT fm.*,
                    f.RUN_ID,
                    f.PRECURSOR_ID,
                    f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE,
                    p.DECOY,
                    COALESCE(ts.TRANSITION_COUNT, 0) AS TRANSITION_COUNT,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM FEATURE_MS2 fm
                INNER JOIN FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
                LEFT JOIN (
                    SELECT tpm.PRECURSOR_ID,
                        COUNT(*) AS TRANSITION_COUNT
                    FROM TRANSITION_PRECURSOR_MAPPING tpm
                    INNER JOIN TRANSITION t ON tpm.TRANSITION_ID = t.ID
                    WHERE t.DETECTING = 1
                    GROUP BY tpm.PRECURSOR_ID
                ) ts ON f.PRECURSOR_ID = ts.PRECURSOR_ID
                WHERE 1=1{run_filter_clause}
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
            """
        else:
            query = f"""
                SELECT fm.*,
                    f.RUN_ID,
                    f.PRECURSOR_ID,
                    f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE,
                    p.DECOY,
                    g.DECOY_PEPTIDE,
                    g.DECOY_GLYCAN,
                    COALESCE(ts.TRANSITION_COUNT, 0) AS TRANSITION_COUNT,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM FEATURE_MS2 fm
                INNER JOIN FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
                INNER JOIN (
                    SELECT pgm.PRECURSOR_ID,
                        gp.DECOY_PEPTIDE,
                        gp.DECOY_GLYCAN
                    FROM PRECURSOR_GLYCOPEPTIDE_MAPPING pgm
                    INNER JOIN GLYCOPEPTIDE gp ON pgm.GLYCOPEPTIDE_ID = gp.ID
                ) g ON f.PRECURSOR_ID = g.PRECURSOR_ID
                LEFT JOIN (
                    SELECT tpm.PRECURSOR_ID,
                        COUNT(*) AS TRANSITION_COUNT
                    FROM TRANSITION_PRECURSOR_MAPPING tpm
                    INNER JOIN TRANSITION t ON tpm.TRANSITION_ID = t.ID
                    WHERE t.DETECTING = 1
                    GROUP BY tpm.PRECURSOR_ID
                ) ts ON f.PRECURSOR_ID = ts.PRECURSOR_ID
                WHERE 1=1{run_filter_clause}
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
            """

        df = pd.read_sql_query(query, con)

        if self.level == "ms1ms2":
            ms1_df = pd.read_sql_query(
                f"""
                SELECT fm1.*
                FROM FEATURE_MS1 fm1
                INNER JOIN FEATURE f ON fm1.FEATURE_ID = f.ID
                WHERE 1=1{run_filter_clause}
                """,
                con,
            )
            ms1_scores = [c for c in ms1_df.columns if c.startswith("VAR_")]
            ms1_df = ms1_df[["FEATURE_ID"] + ms1_scores]
            ms1_df.columns = ["FEATURE_ID"] + [
                "VAR_MS1_" + s.split("VAR_")[1] for s in ms1_scores
            ]
            df = pd.merge(df, ms1_df, how="left", on="FEATURE_ID")

        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _fetch_ms1_features_sqlite(self, con):
        rc = self.config.runner
        glyco = rc.glyco
        ipf_max_rank = rc.ipf_max_peakgroup_rank

        if not check_sqlite_table(con, "FEATURE_MS1"):
            raise click.ClickException("MS1-level feature table not present in file.")

        run_filter_clause = self._get_run_filter_clause("f")

        if not glyco:
            query = f"""
                SELECT fm.*, f.RUN_ID, f.PRECURSOR_ID, f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE, p.DECOY,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM FEATURE_MS1 fm
                INNER JOIN FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
                WHERE 1=1{run_filter_clause}
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
            """
        else:
            if not check_sqlite_table(con, "SCORE_MS2"):
                raise click.ClickException(
                    "MS1-level scoring for glycoform inference requires prior MS2 or MS1MS2-level scoring. "
                    "Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first."
                )

            query = f"""
                SELECT g.DECOY_PEPTIDE, g.DECOY_GLYCAN,
                    fm.*, f.*, p.*,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM FEATURE_MS1 fm
                INNER JOIN FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID
                INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
                INNER JOIN (
                    SELECT pgm.PRECURSOR_ID,
                        gp.DECOY_PEPTIDE,
                        gp.DECOY_GLYCAN
                    FROM PRECURSOR_GLYCOPEPTIDE_MAPPING pgm
                    INNER JOIN GLYCOPEPTIDE gp ON pgm.GLYCOPEPTIDE_ID = gp.ID
                ) g ON f.PRECURSOR_ID = g.PRECURSOR_ID
                WHERE s.RANK <= {ipf_max_rank}{run_filter_clause}
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
            """

        df = pd.read_sql_query(query, con)
        return self._finalize_feature_table(df, rc.ss_main_score)

    def _fetch_transition_features_sqlite(self, con):
        rc = self.config.runner

        if not check_sqlite_table(con, "SCORE_MS2"):
            raise click.ClickException(
                "Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. "
                "Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' first."
            )
        if not check_sqlite_table(con, "FEATURE_TRANSITION"):
            raise click.ClickException(
                "Transition-level feature table not present in file."
            )

        run_filter_clause = self._get_run_filter_clause("f")
        include_mapping_cardinality = rc.transition_score_use_mapping_cardinality
        include_unique_mapping = rc.transition_score_use_unique_mapping
        include_phospho_loss = rc.transition_score_use_phospho_loss
        need_training_unique = rc.transition_training_require_unique_mapping
        need_training_phospho_loss = rc.transition_training_require_phospho_loss
        need_mapping_counts = (
            include_mapping_cardinality
            or include_unique_mapping
            or need_training_unique
        )
        transition_cols = set(get_table_columns(self.infile, "TRANSITION"))
        extra_select_parts = []
        if include_mapping_cardinality:
            extra_select_parts.append(
                "COALESCE(tmc.N_MAPPED_PEPTIDES, 0) AS VAR_MAPPING_CARDINALITY"
            )
        if include_unique_mapping:
            extra_select_parts.append(
                """CASE
                        WHEN COALESCE(tmc.N_MAPPED_PEPTIDES, 0) = 1 THEN 1.0
                        ELSE 0.0
                    END AS VAR_IS_UNIQUE_MAPPING"""
            )
        if need_training_unique:
            extra_select_parts.append(
                """CASE
                        WHEN COALESCE(tmc.N_MAPPED_PEPTIDES, 0) = 1 THEN 1.0
                        ELSE 0.0
                    END AS meta_is_unique_mapping"""
            )
        if include_phospho_loss or need_training_phospho_loss:
            if "ANNOTATION" in transition_cols:
                extra_select_parts.append("tr.ANNOTATION AS TRANSITION_ANNOTATION")
            else:
                extra_select_parts.append("NULL AS TRANSITION_ANNOTATION")
        extra_select_sql = ""
        if extra_select_parts:
            extra_select_sql = ",\n                    " + ",\n                    ".join(extra_select_parts)
        mapping_join_sql = ""
        if need_mapping_counts:
            mapping_join_sql = """
                LEFT JOIN (
                    SELECT
                        TRANSITION_ID,
                        COUNT(DISTINCT PEPTIDE_ID) AS N_MAPPED_PEPTIDES
                    FROM TRANSITION_PEPTIDE_MAPPING
                    GROUP BY TRANSITION_ID
                ) tmc ON ft.TRANSITION_ID = tmc.TRANSITION_ID
            """
        if getattr(self.config, "run_id_filter", None) is not None:
            con.execute("DROP TABLE IF EXISTS temp_run_features")
            con.execute(
                f"""
                CREATE TEMP TABLE temp_run_features AS
                SELECT
                    f.ID AS FEATURE_ID,
                    f.RUN_ID,
                    f.PRECURSOR_ID,
                    f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE
                FROM FEATURE f
                INNER JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID
                INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
                WHERE s.RANK <= {rc.ipf_max_peakgroup_rank}
                AND s.PEP <= {rc.ipf_max_peakgroup_pep}
                AND p.DECOY = 0{run_filter_clause}
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_temp_run_features_feature_id ON temp_run_features (FEATURE_ID)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_temp_run_features_run_precursor ON temp_run_features (RUN_ID, PRECURSOR_ID)"
            )
            query = f"""
                SELECT
                    ft.*{extra_select_sql},
                    tr.DECOY AS DECOY,
                    rf.RUN_ID,
                    rf.PRECURSOR_ID,
                    rf.EXP_RT,
                    rf.PRECURSOR_CHARGE,
                    tr.CHARGE AS PRODUCT_CHARGE,
                    rf.RUN_ID || '_' || ft.FEATURE_ID || '_' || rf.PRECURSOR_ID || '_' || ft.TRANSITION_ID AS GROUP_ID
                FROM temp_run_features rf
                INNER JOIN FEATURE_TRANSITION ft ON ft.FEATURE_ID = rf.FEATURE_ID
                INNER JOIN TRANSITION tr ON ft.TRANSITION_ID = tr.ID
                {mapping_join_sql}
                WHERE ft.VAR_ISOTOPE_OVERLAP_SCORE <= {rc.ipf_max_transition_isotope_overlap}
                AND ft.VAR_LOG_SN_SCORE > {rc.ipf_min_transition_sn}
                ORDER BY rf.RUN_ID, rf.PRECURSOR_ID, rf.EXP_RT, ft.TRANSITION_ID
            """
        else:
            query = f"""
                SELECT
                    ft.*{extra_select_sql},
                    tr.DECOY AS DECOY,
                    f.RUN_ID,
                    f.PRECURSOR_ID,
                    f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE,
                    tr.CHARGE AS PRODUCT_CHARGE,
                    f.RUN_ID || '_' || ft.FEATURE_ID || '_' || f.PRECURSOR_ID || '_' || ft.TRANSITION_ID AS GROUP_ID
                FROM FEATURE_TRANSITION ft
                INNER JOIN FEATURE f ON ft.FEATURE_ID = f.ID
                INNER JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID
                INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
                INNER JOIN TRANSITION tr ON ft.TRANSITION_ID = tr.ID
                {mapping_join_sql}
                WHERE s.RANK <= {rc.ipf_max_peakgroup_rank}
                AND s.PEP <= {rc.ipf_max_peakgroup_pep}
                AND ft.VAR_ISOTOPE_OVERLAP_SCORE <= {rc.ipf_max_transition_isotope_overlap}
                AND ft.VAR_LOG_SN_SCORE > {rc.ipf_min_transition_sn}
                AND p.DECOY = 0
                ORDER BY f.RUN_ID, f.PRECURSOR_ID, f.EXP_RT, ft.TRANSITION_ID
            """
        df = pd.read_sql_query(query, con)
        if include_phospho_loss or need_training_phospho_loss:
            transition_annotation = df["TRANSITION_ANNOTATION"].astype("string")
            phospho_loss = (
                transition_annotation
                .fillna("")
                .str.contains("-H3O4P1", regex=False)
                .astype(float)
            )
            if include_phospho_loss:
                df["VAR_HAS_PHOSPHO_LOSS"] = phospho_loss
            if need_training_phospho_loss:
                df["meta_has_phospho_loss"] = phospho_loss
            df = df.drop(columns=["TRANSITION_ANNOTATION"])
        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _fetch_alignment_features_sqlite(self, con):
        if not check_sqlite_table(con, "FEATURE_MS2_ALIGNMENT"):
            raise click.ClickException(
                "MS2-level feature alignment table not present in file."
            )
        run_filter_clause = self._get_run_filter_clause("fa")
        query = f"""
            SELECT
                fa.ALIGNMENT_ID AS ALIGNMENT_ID, fa.RUN_ID,
                fa.PRECURSOR_ID, fa.ALIGNED_FEATURE_ID AS FEATURE_ID,
                fa.ALIGNED_RT, fa.LABEL AS DECOY,
                fa.XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
                fa.XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE,
                fa.MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE,
                fa.XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,
                fa.XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE_TO_ALL,
                fa.MI_TO_ALL AS VAR_MI_TO_ALL,
                fa.RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION,
                fa.PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO,
                fa.ALIGNED_FEATURE_ID || '_' || fa.PRECURSOR_ID AS GROUP_ID
            FROM FEATURE_MS2_ALIGNMENT fa
            WHERE 1=1{run_filter_clause}
            ORDER BY fa.RUN_ID, fa.PRECURSOR_ID, fa.REFERENCE_RT
        """
        df = pd.read_sql_query(query, con)
        df["DECOY"] = df["DECOY"].map({1: 0, -1: 1})
        return self._finalize_feature_table(df, self.config.runner.ss_main_score)


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

    def __init__(self, config: RunnerIOConfig):
        super().__init__(config)

    def _get_output_tables(self):
        if self.glyco and self.level in ("ms2", "ms1ms2"):
            return ["SCORE_MS2", "SCORE_MS2_PART_PEPTIDE", "SCORE_MS2_PART_GLYCAN"]
        if self.glyco and self.level == "ms1":
            return ["SCORE_MS1", "SCORE_MS1_PART_PEPTIDE", "SCORE_MS1_PART_GLYCAN"]

        table_name = {
            "ms2": "SCORE_MS2",
            "ms1ms2": "SCORE_MS2",
            "ms1": "SCORE_MS1",
            "transition": "SCORE_TRANSITION",
            "alignment": "SCORE_ALIGNMENT",
        }[self.level]
        return [table_name]

    def _drop_output_tables(self, con, tables):
        cur = con.cursor()
        for tbl in tables:
            cur.execute(f"DROP TABLE IF EXISTS {tbl}")
        con.commit()

    def _write_scored_tables(self, df, con):
        if self.glyco and self.level in ("ms2", "ms1ms2"):
            df_main = df[
                [
                    "feature_id",
                    "d_score_combined",
                    "peak_group_rank",
                    "q_value",
                    "pep",
                ]
            ].copy()

            if "h_score" in df.columns:
                df_main["h_score"] = df["h_score"]
                df_main["h0_score"] = df["h0_score"]

            df_main.columns = [c.upper() for c in df_main.columns]
            df_main = df_main.rename(
                columns={"PEAK_GROUP_RANK": "RANK", "D_SCORE_COMBINED": "SCORE"}
            )
            df_main.to_sql("SCORE_MS2", con, index=False, if_exists="append")

            for part in ["peptide", "glycan"]:
                df_part = df[["feature_id", f"d_score_{part}", f"pep_{part}"]].copy()
                df_part.columns = ["FEATURE_ID", "SCORE", "PEP"]
                df_part.to_sql(
                    f"SCORE_MS2_PART_{part.upper()}",
                    con,
                    index=False,
                    if_exists="append",
                )
            return

        if self.glyco and self.level == "ms1":
            df_main = df[
                [
                    "feature_id",
                    "d_score_combined",
                    "peak_group_rank",
                    "q_value",
                    "pep",
                ]
            ].copy()

            if "h_score" in df.columns:
                df_main["h_score"] = df["h_score"]
                df_main["h0_score"] = df["h0_score"]

            df_main.columns = [c.upper() for c in df_main.columns]
            df_main = df_main.rename(
                columns={
                    "PEAK_GROUP_RANK": "RANK",
                    "D_SCORE_COMBINED": "SCORE",
                    "QVALUE": "Q_VALUE",
                }
            )
            df_main.to_sql("SCORE_MS1", con, index=False, if_exists="append")

            for part in ["peptide", "glycan"]:
                df_part = df[["feature_id", f"d_score_{part}", f"pep_{part}"]].copy()
                df_part.columns = ["FEATURE_ID", "SCORE", "PEP"]
                df_part.to_sql(
                    f"SCORE_MS1_PART_{part.upper()}",
                    con,
                    index=False,
                    if_exists="append",
                )
            return

        table_name = self._get_output_tables()[0]
        score_df = self._prepare_score_dataframe(df, self.level, table_name + "_")
        score_df.to_sql(table_name, con, index=False, if_exists="append")

    def save_results(self, result, pi0):
        """
        Save the results to the output file based on the specified level and glyco flag.

        Parameters:
        - result: The result object containing scored tables.
        - pi0: The pi0 value.

        Returns:
        None
        """
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        df = result.scored_tables
        tables = self._get_output_tables()

        # Drop existing tables
        with sqlite3.connect(self.config.outfile) as con:
            self._drop_output_tables(con, tables)
            self._write_scored_tables(df, con)

        logger.success(f"{self.outfile} written.")

        # Save report if statistics are present
        self._write_pdf_report(result, pi0)

    def save_results_incremental(self, scored_table, reset=False):
        if self.glyco:
            raise click.ClickException(
                "Incremental OSW score writing is not supported for glyco scoring."
            )

        if self.infile != self.outfile and reset and not os.path.exists(self.outfile):
            copyfile(self.infile, self.outfile)

        with sqlite3.connect(self.config.outfile) as con:
            if reset:
                self._drop_output_tables(con, self._get_output_tables())
            self._write_scored_tables(scored_table, con)

    def save_scorer(self, scorer):
        if scorer is None:
            return

        raw_blob = pickle.dumps(scorer, protocol=pickle.HIGHEST_PROTOCOL)
        blob = sqlite3.Binary(zlib.compress(raw_blob, level=1))
        with sqlite3.connect(self.outfile) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS PYPROPHET_SCORER (
                    LEVEL TEXT NOT NULL,
                    CLASSIFIER TEXT NOT NULL,
                    SCORER BLOB NOT NULL,
                    PRIMARY KEY (LEVEL, CLASSIFIER)
                )
                """
            )
            con.execute(
                "DELETE FROM PYPROPHET_SCORER WHERE LEVEL = ? AND CLASSIFIER = ?",
                (self.level, self.classifier),
            )
            con.execute(
                "INSERT INTO PYPROPHET_SCORER (LEVEL, CLASSIFIER, SCORER) VALUES (?, ?, ?)",
                (self.level, self.classifier, blob),
            )
            con.commit()

    def save_weights(self, weights):
        """
        Save the weights to a SQLite database based on the classifier type.
        If classifier is "LDA" or "SVM", weights are saved to PYPROPHET_WEIGHTS table.
        If classifier is "XGBoost", weights are saved to PYPROPHET_XGB or GLYCOPEPTIDEPROPHET_XGB table based on glyco and level.
        """
        if self.classifier in ("LDA", "SVM"):
            weights["level"] = self.level
            con = sqlite3.connect(self.outfile)

            c = con.cursor()
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                c.execute(
                    'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="GLYCOPEPTIDEPROPHET_WEIGHTS";'
                )
                if c.fetchone()[0] == 1:
                    c.execute(
                        'DELETE FROM GLYCOPEPTIDEPROPHET_WEIGHTS WHERE LEVEL =="%s"'
                        % self.level
                    )
            else:
                c.execute(
                    'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_WEIGHTS";'
                )
                if c.fetchone()[0] == 1:
                    c.execute(
                        'DELETE FROM PYPROPHET_WEIGHTS WHERE LEVEL =="%s"' % self.level
                    )
            c.close()

            # print(weights)

            weights.to_sql("PYPROPHET_WEIGHTS", con, index=False, if_exists="append")
            con.commit()

        elif self.classifier == "XGBoost":
            con = sqlite3.connect(self.outfile)

            c = con.cursor()
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                c.execute(
                    'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="GLYCOPEPTIDEPROPHET_XGB";'
                )
                if c.fetchone()[0] == 1:
                    c.execute(
                        'DELETE FROM GLYCOPEPTIDEPROPHET_XGB WHERE LEVEL =="%s"'
                        % self.level
                    )
                else:
                    c.execute(
                        "CREATE TABLE GLYCOPEPTIDEPROPHET_XGB (level TEXT, xgb BLOB)"
                    )

                c.execute(
                    "INSERT INTO GLYCOPEPTIDEPROPHET_XGB VALUES(?, ?)",
                    [self.level, pickle.dumps(weights)],
                )
            else:
                c.execute(
                    'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_XGB";'
                )
                if c.fetchone()[0] == 1:
                    c.execute(
                        'DELETE FROM PYPROPHET_XGB WHERE LEVEL =="%s"' % self.level
                    )
                else:
                    c.execute("CREATE TABLE PYPROPHET_XGB (level TEXT, xgb BLOB)")

                c.execute(
                    "INSERT INTO PYPROPHET_XGB VALUES(?, ?)",
                    [self.level, pickle.dumps(weights)],
                )
            con.commit()
            c.close()
