import os

import click
import duckdb
import pandas as pd
from loguru import logger

from ..._config import RunnerIOConfig
from .._base import BaseSplitParquetReader, BaseSplitParquetWriter
from ..util import _ensure_pyarrow, get_parquet_column_names

pa, _, _ = _ensure_pyarrow()


class SplitParquetReader(BaseSplitParquetReader):
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

    def __init__(self, config: RunnerIOConfig):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        """
        Reads and processes data from a DuckDB connection to generate a final feature table based on the specified level and main score.

        Returns:
            pd.DataFrame: Final feature table with the specified main score.
        """
        con = duckdb.connect()
        self._init_duckdb_views(con)

        ss_main_score = self.config.runner.ss_main_score
        feature_table = self._fetch_feature_table(con)

        if self.level == "ms1ms2":
            ms1_cols = self._get_columns_by_prefix(
                "precursors_features.parquet", "FEATURE_MS1_VAR"
            )
            feature_table = self._merge_ms1ms2_features(con, feature_table, ms1_cols)

        con.close()

        return self._finalize_feature_table(feature_table, ss_main_score)

    def _fetch_feature_table(self, con):
        if self.level in ("ms2", "ms1ms2"):
            return self._fetch_ms2_features(
                con,
                self._get_columns_by_prefix(
                    "precursors_features.parquet", "FEATURE_MS2_"
                ),
            )
        elif self.level == "ms1":
            return self._fetch_ms1_features(
                con,
                self._get_columns_by_prefix(
                    "precursors_features.parquet", "FEATURE_MS1_"
                ),
            )
        elif self.level == "transition":
            if not self._get_columns_by_prefix(
                "precursors_features.parquet", "SCORE_MS2_"
            ):
                raise click.ClickException(
                    "Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first."
                )
            # if not os.path.exists(
            #     os.path.join(self.infile, "transition_features.parquet")
            # ):
            #     raise click.ClickException("Transition-level feature table not found.")
            return self._fetch_transition_features(
                con,
                self._get_columns_by_prefix(
                    "transition_features.parquet", "FEATURE_TRANSITION_VAR"
                ),
            )
        elif self.level == "alignment":
            return self._fetch_alignment_features(
                con, self._get_columns_by_prefix("feature_alignment.parquet", "VAR_")
            )
        else:
            raise click.ClickException(
                "Unsupported level for reading semi-supervised input."
            )

    def _fetch_ms2_features(self, con, feature_cols):
        cols_sql = ", ".join([f"p.{col}" for col in feature_cols])
        query = f"""SELECT p.RUN_ID, p.PRECURSOR_ID, p.PROTEIN_ID, p.PRECURSOR_CHARGE, p.FEATURE_ID,
                        p.EXP_RT, p.PRECURSOR_DECOY AS DECOY, {cols_sql},
                        COALESCE(t.TRANSITION_COUNT, 0) AS TRANSITION_COUNT,
                        p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
                    FROM precursors p
                    LEFT JOIN (
                        SELECT PRECURSOR_ID, COUNT(*) AS TRANSITION_COUNT
                        FROM (
                            SELECT DISTINCT PRECURSOR_ID, TRANSITION_ID
                            FROM transition
                            WHERE TRANSITION_DETECTING = 1
                        ) sub
                        GROUP BY PRECURSOR_ID
                    ) t ON p.PRECURSOR_ID = t.PRECURSOR_ID
                    ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT"""
        df = (
            con.execute(query)
            .pl()
            .rename({col: col.replace("FEATURE_MS2_", "") for col in feature_cols})
        )

        df = self._collapse_protein_ids(df)

        return df.to_pandas()

    def _fetch_ms1_features(self, con, feature_cols):
        cols_sql = ", ".join([f"p.{col}" for col in feature_cols])
        query = f"""SELECT DISTINCT p.RUN_ID, p.PRECURSOR_ID, p.PROTEIN_ID, p.PRECURSOR_CHARGE, p.FEATURE_ID,
                        p.EXP_RT, p.PRECURSOR_DECOY AS DECOY, {cols_sql},
                        p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
                    FROM precursors p
                    WHERE p.RUN_ID IS NOT NULL
                    ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT"""
        df = (
            con.execute(query)
            .pl()
            .rename({col: col.replace("FEATURE_MS1_", "") for col in feature_cols})
        )
        df = self._collapse_protein_ids(df)
        return df.to_pandas()

    def _fetch_transition_features(self, con, feature_cols):
        cols_sql = ", ".join([f"t.{col}" for col in feature_cols])
        rc = self.config.runner
        query = f"""SELECT t.TRANSITION_DECOY AS DECOY, t.RUN_ID, t.FEATURE_ID, t.IPF_PEPTIDE_ID, t.TRANSITION_ID, t.FEATURE_TRANSITION_AREA_INTENSITY AS AREA_INTENSITY,
                        {cols_sql}, p.PRECURSOR_CHARGE, t.TRANSITION_CHARGE,
                        p.RUN_ID || '_' || t.FEATURE_ID || '_' || t.PRECURSOR_ID || '_' || t.TRANSITION_ID AS GROUP_ID
                    FROM transition t
                    INNER JOIN precursors p ON t.PRECURSOR_ID = p.PRECURSOR_ID AND t.FEATURE_ID = p.FEATURE_ID
                    WHERE p.SCORE_MS2_PEAK_GROUP_RANK <= {rc.ipf_max_peakgroup_rank}
                        AND p.SCORE_MS2_PEP <= {rc.ipf_max_peakgroup_pep}
                        AND p.PRECURSOR_DECOY = 0
                        AND t.FEATURE_TRANSITION_VAR_ISOTOPE_OVERLAP_SCORE <= {rc.ipf_max_transition_isotope_overlap}
                        AND t.FEATURE_TRANSITION_VAR_LOG_SN_SCORE > {rc.ipf_min_transition_sn}
                    ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT, t.TRANSITION_ID"""
        logger.trace(f"Transition features SQL query: {query}")
        df = (
            con.execute(query)
            .pl()
            .rename(
                {col: col.replace("FEATURE_TRANSITION_", "") for col in feature_cols}
            )
        )
        df = self._collapse_ipf_peptide_ids(df)
        return df.to_pandas()

    def _fetch_alignment_features(self, con, feature_cols):
        cols_sql = ", ".join([f"a.{col}" for col in feature_cols])
        query = f"""SELECT a.ALIGNMENT_ID, a.RUN_ID, a.PRECURSOR_ID, a.FEATURE_ID,
                        a.ALIGNED_RT, a.DECOY, {cols_sql},
                        a.FEATURE_ID || '_' || a.PRECURSOR_ID AS GROUP_ID
                    FROM feature_alignment a
                    ORDER BY a.RUN_ID, a.PRECURSOR_ID, a.REFERENCE_RT"""

        df = con.execute(query).df()
        # Map DECOY to 1 and -1 to 0 and 1
        # arycal saves a label column to indicate 1 as target aligned peaks and -1 as the random/shuffled decoy aligned peak
        df["DECOY"] = df["DECOY"].map({1: 0, -1: 1})

        return df

    def _merge_ms1ms2_features(self, con, df, feature_cols):
        cols_sql = ", ".join([f"p.{col}" for col in feature_cols])
        query = f"SELECT DISTINCT p.FEATURE_ID, {cols_sql} FROM precursors p"
        ms1_df = con.execute(query).df()

        feature_cols = ms1_df.columns.tolist()
        feature_cols.remove("FEATURE_ID")
        ms1_df = ms1_df.rename(
            columns={
                col: col.replace("FEATURE_MS1_VAR_", "VAR_MS1_") for col in feature_cols
            }
        )

        return pd.merge(df, ms1_df, how="left", on="FEATURE_ID")


class SplitParquetWriter(BaseSplitParquetWriter):
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

    def __init__(self, config: RunnerIOConfig):
        super().__init__(config)

    def save_results(self, result, pi0):
        """
        Save the results of scoring to parquet files based on the specified level.

        Parameters:
        - result: The result object containing scored tables.
        - pi0: The pi0 value used in scoring.
        """
        df = result.scored_tables
        level = self.level

        file_map = {
            "ms2": "precursors_features.parquet",
            "ms1ms2": "precursors_features.parquet",
            "ms1": "precursors_features.parquet",
            "transition": "transition_features.parquet",
            "alignment": "feature_alignment.parquet",
        }

        prefix_map = {
            "ms2": "SCORE_MS2_",
            "ms1ms2": "SCORE_MS2_",
            "ms1": "SCORE_MS1_",
            "transition": "SCORE_TRANSITION_",
            "alignment": "SCORE_ALIGNMENT_",
        }

        file_key = file_map.get(level)
        prefix = prefix_map.get(level)

        score_df = self._prepare_score_dataframe(df, level, prefix)

        if self.file_type == "parquet_split_multi" and self.level != "alignment":
            run_dirs = [
                os.path.join(self.outfile, d)
                for d in os.listdir(self.outfile)
                if (d.endswith(".oswpq") or d.endswith(".oswpqd"))
                and os.path.isdir(os.path.join(self.outfile, d))
            ]

            for run_dir in run_dirs:
                file_path = os.path.join(run_dir, file_key)

                if not os.path.exists(file_path):
                    logger.warning(
                        f"Warning: {file_path} does not exist. Skipping.",
                    )
                    continue

                try:
                    con = duckdb.connect()
                    feature_ids = con.execute(
                        f"SELECT FEATURE_ID FROM read_parquet('{file_path}')"
                    ).fetchall()
                    feature_ids = set(f[0] for f in feature_ids)
                    con.close()
                except duckdb.Error as e:
                    logger.error(
                        f"Error reading FEATURE_IDs from {file_path}: {e}",
                    )
                    continue

                subset = score_df[score_df["FEATURE_ID"].isin(feature_ids)]

                if subset.empty:
                    logger.warning(f"No scores for {run_dir}, skipping write.")
                    continue

                existing_cols = get_parquet_column_names(file_path)
                columns_to_keep = self._get_columns_to_keep(existing_cols, prefix)

                self._write_parquet_with_scores(file_path, subset, columns_to_keep)

        else:
            file_path = os.path.join(self.outfile, file_key)

            if not os.path.exists(file_path):
                raise click.ClickException(f"Error: {file_path} not found for writing.")

            existing_cols = get_parquet_column_names(file_path)
            columns_to_keep = self._get_columns_to_keep(existing_cols, prefix)

            self._write_parquet_with_scores(file_path, score_df, columns_to_keep)

        self._write_pdf_report(result, pi0)

    def _write_parquet_with_scores(
        self, target_file: str, df: pd.DataFrame, keep_columns: list[str]
    ):
        # Define columns for scoring
        new_score_cols = [
            col
            for col in df.columns
            if col
            not in (
                "FEATURE_ID",
                "PROTEIN_ID",
                "TRANSITION_ID",
                "IPF_PEPTIDE_ID",
                "ALIGNMENT_ID",
                "DECOY",
            )
        ]
        new_score_sql = ", ".join([f"s.{col}" for col in new_score_cols])

        prefix = "p"
        join_on = "p.FEATURE_ID = s.FEATURE_ID AND p.PROTEIN_ID = s.PROTEIN_ID"
        key_cols = "p.FEATURE_ID"
        if self.level == "transition":
            prefix = "t"
            join_on = "t.FEATURE_ID = s.FEATURE_ID AND t.TRANSITION_ID = s.TRANSITION_ID AND t.IPF_PEPTIDE_ID = s.IPF_PEPTIDE_ID"
            key_cols = "t.FEATURE_ID, t.TRANSITION_ID, t.IPF_PEPTIDE_ID"
        if self.level == "alignment":
            prefix = "a"
            join_on = "a.ALIGNMENT_ID = s.ALIGNMENT_ID AND a.FEATURE_ID = s.FEATURE_ID AND a.DECOY = s.DECOY"
            key_cols = "a.ALIGNMENT_ID, a.FEATURE_ID, a.DECOY"

        table_alias = prefix
        select_old = ", ".join([f"{table_alias}.{col}" for col in keep_columns])

        # Register the DataFrame as a table in DuckDB
        con = duckdb.connect()
        con.register("scores", pa.Table.from_pandas(df))

        # Validate input row entry count and joined entry count remain the same
        self._validate_row_count_after_join(con, target_file, key_cols, join_on, prefix)

        con.execute(
            f"""
            COPY (
                SELECT {select_old}, {new_score_sql}
                FROM read_parquet('{target_file}') {table_alias}
                LEFT JOIN scores s
                ON {join_on}
            ) TO '{target_file}'
            (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11);
            """
        )

        logger.debug(
            f"After appendings scores, {target_file} has {self._get_parquet_row_count(con, target_file)} entries"
        )

        con.close()

        logger.success(f"{target_file} written successfully.")
