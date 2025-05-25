import sys
from shutil import copyfile
import pandas as pd
import pyarrow as pa
import duckdb
import click
from loguru import logger
from ..util import get_parquet_column_names
from .._base import BaseReader, BaseWriter, RowCountMismatchError
from ..._config import RunnerIOConfig


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

    def __init__(self, config: RunnerIOConfig):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        con = duckdb.connect()
        self._init_duckdb_views(con)

        ss_main_score = self.config.runner.ss_main_score
        feature_table = self._fetch_feature_table(con)

        if self.level == "ms1ms2":
            ms1_cols = self._get_columns_by_prefix(self.infile, "FEATURE_MS1_VAR")
            feature_table = self._merge_ms1ms2_features(con, feature_table, ms1_cols)

        return self._finalize_feature_table(feature_table, ss_main_score)

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
                FROM read_parquet({self.infile})
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

    def _get_columns_by_prefix(self, parquet_file, prefix):
        cols = get_parquet_column_names(parquet_file)
        return [c for c in cols if c.startswith(prefix)]

    def _fetch_feature_table(self, con):
        if self.level in ("ms2", "ms1ms2"):
            return self._fetch_ms2_features(
                con,
                self._get_columns_by_prefix(self.infile, "FEATURE_MS2_"),
            )
        elif self.level == "ms1":
            return self._fetch_ms1_features(
                con,
                self._get_columns_by_prefix(self.infile, "FEATURE_MS1_"),
            )
        elif self.level == "transition":
            if not self._get_columns_by_prefix(self.infile, "SCORE_MS2_"):
                raise click.ClickException(
                    "Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first."
                )

            return self._fetch_transition_features(
                con,
                self._get_columns_by_prefix(self.infile, "FEATURE_TRANSITION_VAR"),
            )
        elif self.level == "alignment":
            return self._fetch_alignment_features(
                con, self._get_columns_by_prefix(self.infile, "VAR_")
            )
        else:
            raise click.ClickException(
                "Unsupported level for reading semi-supervised input."
            )

    def _fetch_ms2_features(self, con, feature_cols):
        cols_sql = ", ".join([f"p.{col}" for col in feature_cols])
        cols_sql_inner = ", ".join([f"{col}" for col in feature_cols])
        query = f"""
                SELECT
                    p.RUN_ID,
                    p.PRECURSOR_ID,
                    p.PRECURSOR_CHARGE,
                    p.FEATURE_ID,
                    p.EXP_RT,
                    p.PRECURSOR_DECOY AS DECOY,
                    {cols_sql},
                    COALESCE(t.TRANSITION_COUNT, 0) AS TRANSITION_COUNT,
                    p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
                FROM (
                    SELECT
                        RUN_ID,
                        PRECURSOR_ID,
                        PRECURSOR_CHARGE,
                        FEATURE_ID,
                        EXP_RT,
                        PRECURSOR_DECOY,
                        {cols_sql_inner}
                    FROM data
                    WHERE MODIFIED_SEQUENCE IS NOT NULL
                ) AS p
                LEFT JOIN (
                    SELECT PRECURSOR_ID, COUNT(*) AS TRANSITION_COUNT
                    FROM (
                        SELECT DISTINCT PRECURSOR_ID, TRANSITION_ID
                        FROM data
                        WHERE MODIFIED_SEQUENCE IS NULL
                        AND TRANSITION_DETECTING = 1
                    ) sub
                    GROUP BY PRECURSOR_ID
                ) AS t ON p.PRECURSOR_ID = t.PRECURSOR_ID
                ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT
                """

        df = (
            con.execute(query)
            .pl()
            .rename({col: col.replace("FEATURE_MS2_", "") for col in feature_cols})
        )
        return df.to_pandas()

    def _fetch_ms1_features(self, con, feature_cols):
        cols_sql = ", ".join([f"p.{col}" for col in feature_cols])
        query = f"""SELECT DISTINCT p.RUN_ID, p.PRECURSOR_ID, p.PRECURSOR_CHARGE, p.FEATURE_ID,
                        p.EXP_RT, p.PRECURSOR_DECOY AS DECOY, {cols_sql},
                        p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
                    FROM data p
                    WHERE p.RUN_ID IS NOT NULL
                    ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT"""
        df = (
            con.execute(query)
            .pl()
            .rename({col: col.replace("FEATURE_MS1_", "") for col in feature_cols})
        )
        return df.to_pandas()

    def _fetch_transition_features(self, con, feature_cols):
        cols_sql = ", ".join([f"t.{col}" for col in feature_cols])
        cols_sql_inner = ", ".join([f"{col}" for col in feature_cols])
        rc = self.config.runner
        query = f"""SELECT t.TRANSITION_DECOY AS DECOY, t.FEATURE_ID, t.IPF_PEPTIDE_ID, t.TRANSITION_ID,
                        {cols_sql}, p.PRECURSOR_CHARGE, t.TRANSITION_CHARGE,
                        p.RUN_ID || '_' || t.FEATURE_ID || '_' || t.TRANSITION_ID AS GROUP_ID
                    FROM (
                        SELECT
                            RUN_ID,
                            PRECURSOR_ID,
                            IPF_PEPTIDE_ID,
                            TRANSITION_ID,
                            TRANSITION_DECOY,
                            FEATURE_ID,
                            TRANSITION_CHARGE,
                            {cols_sql_inner}
                        FROM data
                        WHERE MODIFIED_SEQUENCE IS NULL
                    ) AS t
                    LEFT JOIN (
                        SELECT PRECURSOR_ID, PRECURSOR_CHARGE, PRECURSOR_DECOY, RUN_ID, FEATURE_ID, EXP_RT, SCORE_MS2_PEP, SCORE_MS2_PEAK_GROUP_RANK
                        FROM (
                            SELECT DISTINCT PRECURSOR_ID, PRECURSOR_CHARGE, PRECURSOR_DECOY, RUN_ID, FEATURE_ID, EXP_RT, SCORE_MS2_PEP, SCORE_MS2_PEAK_GROUP_RANK
                            FROM data
                            WHERE MODIFIED_SEQUENCE IS NOT NULL
                        ) sub
                    ) AS p ON t.PRECURSOR_ID = p.PRECURSOR_ID AND t.FEATURE_ID = p.FEATURE_ID
                    WHERE p.SCORE_MS2_PEAK_GROUP_RANK <= {rc.ipf_max_peakgroup_rank}
                        AND p.SCORE_MS2_PEP <= {rc.ipf_max_peakgroup_pep}
                        AND p.PRECURSOR_DECOY = 0
                        AND t.FEATURE_TRANSITION_VAR_ISOTOPE_OVERLAP_SCORE <= {rc.ipf_max_transition_isotope_overlap}
                        AND t.FEATURE_TRANSITION_VAR_LOG_SN_SCORE > {rc.ipf_min_transition_sn}
                    ORDER BY t.RUN_ID, t.PRECURSOR_ID, p.EXP_RT, t.TRANSITION_ID"""
        df = (
            con.execute(query)
            .pl()
            .rename(
                {col: col.replace("FEATURE_TRANSITION_", "") for col in feature_cols}
            )
        )
        return df.to_pandas()

    def _fetch_alignment_features(self, con, feature_cols):
        cols_sql = ", ".join([f"a.{col}" for col in feature_cols])
        query = f"""SELECT a.ALIGNMENT_ID, a.RUN_ID, a.PRECURSOR_ID, a.FEATURE_ID,
                        a.ALIGNED_RT, a.DECOY, {cols_sql},
                        a.FEATURE_ID || '_' || a.PRECURSOR_ID AS GROUP_ID
                    FROM data a
                    ORDER BY a.RUN_ID, a.PRECURSOR_ID, a.REFERENCE_RT"""
        df = con.execute(query).pl().to_pandas()
        # Map DECOY to 1 and -1 to 0 and 1
        # arycal saves a label column to indicate 1 as target aligned peaks and -1 as the random/shuffled decoy aligned peak
        df["DECOY"] = df["DECOY"].map({1: 0, -1: 1})

        return df

    def _merge_ms1ms2_features(self, con, df, feature_cols):
        cols_sql = ", ".join([f"p.{col}" for col in feature_cols])
        query = f"SELECT DISTINCT p.FEATURE_ID, {cols_sql} FROM data p"
        ms1_df = con.execute(query).df()
        return pd.merge(df, ms1_df, how="left", on="FEATURE_ID")


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

    def __init__(self, config: RunnerIOConfig):
        super().__init__(config)

    def save_results(self, result, pi0):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        df = result.scored_tables
        level = self.level
        prefix_map = {
            "ms2": "SCORE_MS2_",
            "ms1ms2": "SCORE_MS2_",
            "ms1": "SCORE_MS1_",
            "transition": "SCORE_TRANSITION_",
            "alignment": "SCORE_ALIGNMENT_",
        }
        prefix = prefix_map[level]
        target_file = self.outfile

        score_df = self._prepare_score_dataframe(df, level, prefix)
        existing_cols = get_parquet_column_names(target_file)
        columns_to_keep = self._get_columns_to_keep(existing_cols, prefix)

        self._write_parquet_with_scores(target_file, score_df, columns_to_keep)

        self._write_pdf_report(result, pi0)

    def _write_parquet_with_scores(self, target_file, df, keep_columns):
        new_score_cols = [
            col
            for col in df.columns
            if col not in ("FEATURE_ID", "TRANSITION_ID", "ALIGNMENT_ID", "DECOY")
        ]
        new_score_sql = ", ".join([f"s.{col}" for col in new_score_cols])

        if self.level == "transition":
            prefix = "t"
            join_on = "t.FEATURE_ID = s.FEATURE_ID AND t.TRANSITION_ID = s.TRANSITION_ID AND t.IPF_PEPTIDE_ID = s.IPF_PEPTIDE_ID"
            key_cols = "t.FEATURE_ID, t.TRANSITION_ID, t.IPF_PEPTIDE_ID"
        elif self.level == "alignment":
            prefix = "a"
            join_on = "a.ALIGNMENT_ID = s.ALIGNMENT_ID AND a.FEATURE_ID = s.FEATURE_ID AND a.DECOY = s.DECOY"
            key_cols = "a.ALIGNMENT_ID, a.FEATURE_ID"
        else:
            prefix = "p"
            join_on = "p.FEATURE_ID = s.FEATURE_ID"
            key_cols = "p.FEATURE_ID"

        select_old = ", ".join([f"{prefix}.{col}" for col in keep_columns])
        con = duckdb.connect()
        con.register("scores", pa.Table.from_pandas(df))

        # Validate input row entry count and joined entry count remain the same
        self._validate_row_count_after_join(con, target_file, key_cols, join_on, prefix)

        con.execute(
            f"""
            COPY (
                SELECT {select_old}, {new_score_sql}
                FROM read_parquet('{target_file}') {prefix}
                LEFT JOIN scores s ON {join_on}
            ) TO '{target_file}' (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11)
            """
        )

        logger.debug(
            f"After appendings scores, {target_file} has {self._get_parquet_row_count(con, target_file)} entries"
        )

        con.close()
        logger.success(f"{target_file} written.")
