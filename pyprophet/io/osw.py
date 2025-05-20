import os
import pickle
from shutil import copyfile
import sqlite3
import duckdb
import pandas as pd
import click
from ._base import BaseReader, BaseWriter, BaseIOConfig
from .._config import RunnerIOConfig, IPFIOConfig, LevelContextIOConfig
from .util import check_sqlite_table, check_duckdb_table
from ..report import save_report
from ..glyco.report import save_report as save_report_glyco


class OSWReader(BaseReader):
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
        self._create_indexes()
        try:
            con = duckdb.connect()
            con.execute("INSTALL sqlite_scanner;")
            con.execute("LOAD sqlite_scanner;")
            con.execute(f"ATTACH DATABASE '{self.infile}' AS osw (TYPE sqlite);")
            return self._read_using_duckdb(con)
        except ModuleNotFoundError as e:
            click.echo(
                click.style(
                    f"Warn: DuckDB sqlite_scanner failed, falling back to SQLite. Reason: {e}",
                    fg="yellow",
                )
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
                "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);",
                "CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);",
            ]

            for stmt in index_statements:
                try:
                    sqlite_con.execute(stmt)
                except sqlite3.OperationalError as e:
                    click.echo(
                        click.style(
                            f"Warn: SQLite index creation failed: {e}", fg="yellow"
                        )
                    )

            sqlite_con.commit()
            sqlite_con.close()

        except Exception as e:
            raise click.ClickException(
                f"Failed to create indexes via SQLite fallback: {e}"
            )

    def _read_using_duckdb(self, con):
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

    def _fetch_ms2_features_duckdb(self, con):
        if not check_duckdb_table(con, "main", "FEATURE_MS2"):
            raise click.ClickException(
                f"MS2-level feature table not present in file.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )

        if self.glyco:
            con.execute(
                """
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
                """
            )
        else:
            con.execute(
                """
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
                """
            )

        df = con.execute(
            "SELECT * FROM ms2_table ORDER BY RUN_ID, PRECURSOR_ID, EXP_RT"
        ).fetchdf()
        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _fetch_ms1_features_duckdb(self, con):
        if not check_duckdb_table(con, "main", "FEATURE_MS1"):
            raise click.ClickException(
                f"MS1-level feature table not present in file.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )

        rc = self.config.runner
        glyco = rc.glyco
        ipf_max_rank = rc.ipf_max_peakgroup_rank

        if not glyco:
            con.execute(
                """
                CREATE OR REPLACE VIEW ms1_table AS
                SELECT fm.*, f.RUN_ID, f.PRECURSOR_ID, f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE, p.DECOY,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM osw.FEATURE_MS1 fm
                INNER JOIN osw.FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN osw.PRECURSOR p ON f.PRECURSOR_ID = p.ID
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
                """
            )
        else:
            if not check_duckdb_table(con, "main", "SCORE_MS2"):
                raise click.ClickException(
                    f"MS1-level scoring for glycoform inference requires prior MS2 or MS1MS2-level scoring. "
                    "Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first.level\nTable Info:\n{self._fetch_tables_duckdb(con)}"
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
                WHERE s.RANK <= {ipf_max_rank}
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
                """
            )

        df = con.execute("SELECT * FROM ms1_table").fetchdf()
        return self._finalize_feature_table(df, rc.ss_main_score)

    def _fetch_transition_features_duckdb(self, con):
        if not check_duckdb_table(con, "main", "SCORE_MS2"):
            raise click.ClickException(
                f"Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. "
                "Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' first.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )

        if not check_duckdb_table(con, "main", "FEATURE_TRANSITION"):
            raise click.ClickException(
                f"Transition-level feature table not present in file.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )

        rc = self.config.runner
        con.execute(
            f"""
            CREATE OR REPLACE VIEW transition_table AS
            SELECT 
                ft.*,
                t.DECOY AS DECOY,
                f.RUN_ID,
                f.PRECURSOR_ID,
                f.EXP_RT,
                p.CHARGE AS PRECURSOR_CHARGE,
                t.PRODUCT_CHARGE,
                f.RUN_ID || '_' || ft.FEATURE_ID || '_' || f.PRECURSOR_ID || '_' || ft.TRANSITION_ID AS GROUP_ID
            FROM osw.FEATURE_TRANSITION ft
            INNER JOIN osw.FEATURE f ON ft.FEATURE_ID = f.ID
            INNER JOIN osw.SCORE_MS2 s ON f.ID = s.FEATURE_ID
            INNER JOIN osw.PRECURSOR p ON f.PRECURSOR_ID = p.ID
            INNER JOIN osw.TRANSITION t ON ft.TRANSITION_ID = t.ID
            WHERE s.RANK <= {rc.ipf_max_peakgroup_rank}
            AND s.PEP <= {rc.ipf_max_peakgroup_pep}
            AND ft.VAR_ISOTOPE_OVERLAP_SCORE <= {rc.ipf_max_transition_isotope_overlap}
            AND ft.VAR_LOG_SN_SCORE > {rc.ipf_min_transition_sn}
            AND p.DECOY = 0
            """
        )
        df = con.execute(
            """
            SELECT * 
            FROM transition_table 
            ORDER BY RUN_ID, PRECURSOR_ID, EXP_RT, TRANSITION_ID
            """
        ).fetchdf()

        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _fetch_alignment_features_duckdb(self, con):
        if not check_duckdb_table(con, "main", "FEATURE_MS2_ALIGNMENT"):
            raise click.ClickException(
                f"MS2-level feature alignment table not present in file.\nTable Info:\n{self._fetch_tables_duckdb(con)}"
            )
        con.execute(
            """
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

        if not self.glyco:
            query = """
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
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
            """
        else:
            query = """
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
                ORDER BY f.RUN_ID, p.ID, f.EXP_RT
            """

        df = pd.read_sql_query(query, con)
        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _fetch_ms1_features_sqlite(self, con):
        rc = self.config.runner
        glyco = rc.glyco
        ipf_max_rank = rc.ipf_max_peakgroup_rank

        if not check_sqlite_table(con, "FEATURE_MS1"):
            raise click.ClickException("MS1-level feature table not present in file.")

        if not glyco:
            query = """
                SELECT fm.*, f.RUN_ID, f.PRECURSOR_ID, f.EXP_RT,
                    p.CHARGE AS PRECURSOR_CHARGE, p.DECOY,
                    f.RUN_ID || '_' || f.PRECURSOR_ID AS GROUP_ID
                FROM FEATURE_MS1 fm
                INNER JOIN FEATURE f ON fm.FEATURE_ID = f.ID
                INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
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
                WHERE s.RANK <= {ipf_max_rank}
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

        query = f"""
            SELECT 
                ft.*,
                t.DECOY AS DECOY,
                f.RUN_ID,
                f.PRECURSOR_ID,
                f.EXP_RT,
                p.CHARGE AS PRECURSOR_CHARGE,
                t.PRODUCT_CHARGE,
                f.RUN_ID || '_' || ft.FEATURE_ID || '_' || f.PRECURSOR_ID || '_' || ft.TRANSITION_ID AS GROUP_ID
            FROM FEATURE_TRANSITION ft
            INNER JOIN FEATURE f ON ft.FEATURE_ID = f.ID
            INNER JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID
            INNER JOIN PRECURSOR p ON f.PRECURSOR_ID = p.ID
            INNER JOIN TRANSITION t ON ft.TRANSITION_ID = t.ID
            WHERE s.RANK <= {rc.ipf_max_peakgroup_rank}
            AND s.PEP <= {rc.ipf_max_peakgroup_pep}
            AND ft.VAR_ISOTOPE_OVERLAP_SCORE <= {rc.ipf_max_transition_isotope_overlap}
            AND ft.VAR_LOG_SN_SCORE > {rc.ipf_min_transition_sn}
            AND p.DECOY = 0
            ORDER BY f.RUN_ID, f.PRECURSOR_ID, f.EXP_RT, ft.TRANSITION_ID
        """
        df = pd.read_sql_query(query, con)
        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _fetch_alignment_features_sqlite(self, con):
        if not check_sqlite_table(con, "FEATURE_MS2_ALIGNMENT"):
            raise click.ClickException(
                "MS2-level feature alignment table not present in file."
            )
        query = """
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
            ORDER BY fa.RUN_ID, fa.PRECURSOR_ID, fa.REFERENCE_RT
        """
        df = pd.read_sql_query(query, con)
        df["DECOY"] = df["DECOY"].map({1: 0, -1: 1})
        return self._finalize_feature_table(df, self.config.runner.ss_main_score)

    def _read_for_ipf(self):
        raise NotImplementedError

    def _read_for_context_level(self):
        raise NotImplementedError


class OSWWriter(BaseWriter):
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
        if isinstance(self.config, RunnerIOConfig):
            return self._save_semi_supervised_results(result, pi0)
        elif isinstance(self.config, IPFIOConfig):
            return self._save_ipf_results(result)
        elif isinstance(self.config, LevelContextIOConfig):
            return self._save_context_level_results(result)
        else:
            raise NotImplementedError(
                f"Mode {self.config.mode} is not supported in OSWWriter."
            )

    def _save_semi_supervised_results(self, result, pi0):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        df = result.scored_tables
        level = self.level
        glyco = self.glyco

        # Determine output table(s)
        if glyco and level in ("ms2", "ms1ms2"):
            tables = ["SCORE_MS2", "SCORE_MS2_PART_PEPTIDE", "SCORE_MS2_PART_GLYCAN"]
        elif glyco and level == "ms1":
            tables = ["SCORE_MS1", "SCORE_MS1_PART_PEPTIDE", "SCORE_MS1_PART_GLYCAN"]
        else:
            tables = {
                "ms2": "SCORE_MS2",
                "ms1ms2": "SCORE_MS2",
                "ms1": "SCORE_MS1",
                "transition": "SCORE_TRANSITION",
                "alignment": "SCORE_ALIGNMENT",
            }[level]

            if isinstance(tables, str):
                tables = [tables]

        # Drop existing tables
        with sqlite3.connect(self.config.outfile) as con:
            cur = con.cursor()
            for tbl in tables:
                cur.execute(f"DROP TABLE IF EXISTS {tbl}")
            con.commit()

            # Prepare data for writing
            if glyco and level in ("ms2", "ms1ms2"):
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
                df_main.to_sql("SCORE_MS2", con, index=False)

                # Write peptide/glycan part scores
                for part in ["peptide", "glycan"]:
                    df_part = df[
                        ["feature_id", f"d_score_{part}", f"pep_{part}"]
                    ].copy()
                    df_part.columns = ["FEATURE_ID", "SCORE", "PEP"]
                    df_part.to_sql(f"SCORE_MS2_PART_{part.upper()}", con, index=False)

            elif glyco and level == "ms1":
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
                df_main.to_sql("SCORE_MS1", con, index=False)

                for part in ["peptide", "glycan"]:
                    df_part = df[
                        ["feature_id", f"d_score_{part}", f"pep_{part}"]
                    ].copy()
                    df_part.columns = ["FEATURE_ID", "SCORE", "PEP"]
                    df_part.to_sql(f"SCORE_MS1_PART_{part.upper()}", con, index=False)

            else:
                # Regular MS1, MS2, transition, or alignment
                table_name = tables[0]
                score_df = self._prepare_score_dataframe(df, level, table_name + "_")
                score_df.to_sql(table_name, con, index=False)

        click.echo(f"Info: {self.outfile} written.")

        # Save report if statistics are present
        self._write_pdf_report_if_present(result, pi0)

    def _save_ipf_results(self, result):
        raise NotImplementedError

    def _save_context_level_results(self, result):
        raise NotImplementedError

    def save_weights(self, weights):
        if self.classifier == "LDA":
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
