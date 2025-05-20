import os
from shutil import copyfile
import pandas as pd
import pyarrow as pa
import duckdb
import click

from ._base import BaseReader, BaseWriter, BaseIOConfig
from .util import get_parquet_column_names
from .._config import RunnerIOConfig, IPFIOConfig, LevelContextIOConfig


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

    def __init__(self, config: BaseIOConfig):
        super().__init__(config)

        # validate required files
        required_files = [
            "precursors_features.parquet",
            "transition_features.parquet",
        ]
        for file in required_files:
            if not os.path.exists(os.path.join(config.infile, file)):
                click.echo(click.style(f"Error: Couldn't find: {file}", fg="red"))
                raise click.ClickException(
                    click.style(
                        f"{file} is required for processing. Please check the input parquet hive directory.",
                        fg="red",
                    )
                )

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
        con = duckdb.connect()
        self._init_duckdb_views(con)

        ss_main_score = self.config.runner.ss_main_score
        feature_table = self._fetch_feature_table(con)

        if self.level == "ms1ms2":
            ms1_cols = self._get_columns_by_prefix(
                "precursors_features.parquet", "FEATURE_MS1_VAR"
            )
            feature_table = self._merge_ms1ms2_features(con, feature_table, ms1_cols)

        return self._finalize_feature_table(feature_table, ss_main_score)

    def _init_duckdb_views(self, con):
        for name in ["precursors", "transition", "feature_alignment"]:
            if name == "feature_alignment":
                path = os.path.join(self.infile, "feature_alignment.parquet")
            else:
                path = os.path.join(self.infile, f"{name}_features.parquet")
            if os.path.exists(path):
                con.execute(
                    f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')"
                )

    def _get_columns_by_prefix(self, parquet_file, prefix):
        cols = get_parquet_column_names(os.path.join(self.infile, parquet_file))
        return [c for c in cols if c.startswith(prefix)]

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
            if not os.path.exists(
                os.path.join(self.infile, "transition_features.parquet")
            ):
                raise click.ClickException("Transition-level feature table not found.")
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
        query = f"""SELECT p.RUN_ID, p.PRECURSOR_ID, p.PRECURSOR_CHARGE, p.FEATURE_ID,
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
        return df.to_pandas()

    def _fetch_ms1_features(self, con, feature_cols):
        cols_sql = ", ".join([f"p.{col}" for col in feature_cols])
        query = f"""SELECT p.RUN_ID, p.PRECURSOR_ID, p.PRECURSOR_CHARGE, p.FEATURE_ID,
                        p.EXP_RT, p.PRECURSOR_DECOY AS DECOY, {cols_sql},
                        p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
                    FROM precursors p
                    ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT"""
        df = (
            con.execute(query)
            .pl()
            .rename({col: col.replace("FEATURE_MS1_", "") for col in feature_cols})
        )
        return df.to_pandas()

    def _fetch_transition_features(self, con, feature_cols):
        cols_sql = ", ".join([f"t.{col}" for col in feature_cols])
        rc = self.config.runner
        query = f"""SELECT t.TRANSITION_DECOY AS DECOY, t.FEATURE_ID, t.TRANSITION_ID,
                        {cols_sql}, p.PRECURSOR_CHARGE, t.TRANSITION_CHARGE,
                        p.RUN_ID || '_' || t.FEATURE_ID || '_' || t.TRANSITION_ID AS GROUP_ID
                    FROM transition t
                    INNER JOIN precursors p ON t.PRECURSOR_ID = p.PRECURSOR_ID AND t.FEATURE_ID = p.FEATURE_ID
                    WHERE p.SCORE_MS2_PEAK_GROUP_RANK <= {rc.ipf_max_peakgroup_rank}
                        AND p.SCORE_MS2_PEP <= {rc.ipf_max_peakgroup_pep}
                        AND p.PRECURSOR_DECOY = 0
                        AND t.FEATURE_TRANSITION_VAR_ISOTOPE_OVERLAP_SCORE <= {rc.ipf_max_transition_isotope_overlap}
                        AND t.FEATURE_TRANSITION_VAR_LOG_SN_SCORE > {rc.ipf_min_transition_sn}
                    ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT, t.TRANSITION_ID"""
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
                    FROM feature_alignment a
                    ORDER BY a.RUN_ID, a.PRECURSOR_ID, a.REFERENCE_RT"""
        df = con.execute(query).pl().to_pandas()
        # Map DECOY to 1 and -1 to 0 and 1
        # arycal saves a label column to indicate 1 as target aligned peaks and -1 as the random/shuffled decoy aligned peak
        df["DECOY"] = df["DECOY"].map({1: 0, -1: 1})

        return df

    def _merge_ms1ms2_features(self, con, df, feature_cols):
        cols_sql = ", ".join([f"p.{col}" for col in feature_cols])
        query = f"SELECT p.FEATURE_ID, {cols_sql} FROM precursors p"
        ms1_df = con.execute(query).df()
        return pd.merge(df, ms1_df, how="left", on="FEATURE_ID")

    def _read_for_ipf(self):
        raise NotImplementedError

    def _read_for_context_level(self):
        raise NotImplementedError


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

    def __init__(self, config: BaseIOConfig):
        super().__init__(config)

        # validate required files
        required_files = [
            "precursors_features.parquet",
            "transition_features.parquet",
        ]
        for file in required_files:
            if not os.path.exists(os.path.join(config.infile, file)):
                click.echo(click.style(f"Error: Couldn't find: {file}", fg="red"))
                raise click.ClickException(
                    click.style(
                        f"An existing {file} is required for writing results to. Please check the input parquet hive directory.",
                        fg="red",
                    )
                )

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
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

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
        file_path = os.path.join(self.outfile, file_key)

        score_df = self._prepare_score_dataframe(df, level, prefix)

        existing_cols = get_parquet_column_names(file_path)
        columns_to_keep = self._get_columns_to_keep(existing_cols, prefix)

        self._write_parquet_with_scores(file_path, score_df, columns_to_keep)

        self._write_pdf_report_if_present(result, pi0)

    def _write_parquet_with_scores(
        self, target_file: str, df: pd.DataFrame, keep_columns: list[str]
    ):
        new_score_cols = [
            col
            for col in df.columns
            if col not in ("FEATURE_ID", "TRANSITION_ID", "ALIGNMENT_ID", "DECOY")
        ]
        new_score_sql = ", ".join([f"s.{col}" for col in new_score_cols])

        prefix = "p"
        join_on = "p.FEATURE_ID = s.FEATURE_ID"
        if self.level == "transition":
            prefix = "t"
            join_on = (
                "t.FEATURE_ID = s.FEATURE_ID AND t.TRANSITION_ID = s.TRANSITION_ID"
            )
        if self.level == "alignment":
            prefix = "a"
            join_on = "a.ALIGNMENT_ID = s.ALIGNMENT_ID AND a.FEATURE_ID = s.FEATURE_ID AND a.DECOY = s.DECOY"

        table_alias = prefix
        select_old = ", ".join([f"{table_alias}.{col}" for col in keep_columns])

        con = duckdb.connect()
        con.register("scores", pa.Table.from_pandas(df))

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
        click.echo(f"Info: {target_file} written.")

    def _save_ipf_results(self, result):
        raise NotImplementedError

    def _save_context_level_results(self, result):
        raise NotImplementedError
