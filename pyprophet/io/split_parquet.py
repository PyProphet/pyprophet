import os
from shutil import copyfile
import pickle
import pandas as pd
import pyarrow as pa
import duckdb
import click

from ._base import BaseReader, BaseWriter, BaseIOConfig
from .._config import RunnerIOConfig, IPFIOConfig, LevelContextIOConfig
from ..data_handling import get_parquet_column_names
from ..report import save_report


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
        ss_main_score = self.config.ss_main_score

        precursor_file = os.path.join(self.infile, "precursors_features.parquet")
        transition_file = os.path.join(self.infile, "transition_features.parquet")
        alignment_file = os.path.join(self.infile, "feature_alignment.parquet")

        all_precursor_column_names = get_parquet_column_names(precursor_file)
        all_transition_column_names = get_parquet_column_names(transition_file)

        if self.level == "alignment":
            if os.path.exists(alignment_file):
                all_alignment_column_names = get_parquet_column_names(alignment_file)
            else:
                click.echo(
                    click.style(f"Error: Couldn't find: {alignment_file}", fg="red")
                )
                raise click.ClickException(
                    click.style(
                        "Alignment-level features are not present in the input parquet directory. Please run ARYCAL first to perform alignment of features. https://github.com/singjc/arycal",
                        fg="red",
                    )
                )

        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW precursors AS SELECT * FROM read_parquet('{precursor_file}')"
        )
        if (
            os.path.exists(transition_file)
            and os.path.basename(transition_file) == "transition_features.parquet"
        ):
            con.execute(
                f"CREATE VIEW transitions AS SELECT * FROM read_parquet('{transition_file}')"
            )

        if (
            os.path.exists(alignment_file)
            and os.path.basename(alignment_file) == "feature_alignment.parquet"
        ):
            con.execute(
                f"CREATE VIEW alignment_features AS SELECT * FROM read_parquet('{alignment_file}')"
            )

        if self.level in ("ms2", "ms1ms2"):
            if not any(
                [col.startswith("FEATURE_MS2_") for col in all_precursor_column_names]
            ):
                raise click.ClickException(
                    "MS2-level feature columns are not present in precursors_features.parquet file."
                )

            # Filter columes names for FEATURE_MS2_
            feature_ms2_cols = [
                col
                for col in all_precursor_column_names
                if col.startswith("FEATURE_MS2_")
            ]
            # prepare feature ms2 columns for sql query
            feature_ms2_cols_sql = ", ".join([f"p.{col}" for col in feature_ms2_cols])

            query = f"""
            SELECT 
                p.RUN_ID, 
                p.PRECURSOR_ID, 
                p.PRECURSOR_CHARGE, 
                p.FEATURE_ID, 
                p.EXP_RT, 
                p.PRECURSOR_DECOY AS DECOY, 
                {feature_ms2_cols_sql},
                COALESCE(t.TRANSITION_COUNT, 0) AS TRANSITION_COUNT,
                p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
            FROM precursors p
            LEFT JOIN (
                SELECT 
                    PRECURSOR_ID, 
                    COUNT(*) AS TRANSITION_COUNT
                FROM (
                    SELECT DISTINCT PRECURSOR_ID, TRANSITION_ID
                    FROM transitions
                    WHERE TRANSITION_DETECTING = 1
                ) AS sub
                GROUP BY PRECURSOR_ID
            ) AS t
            ON p.PRECURSOR_ID = t.PRECURSOR_ID
            ORDER BY p.RUN_ID, p.PRECURSOR_ID ASC, p.EXP_RT ASC
            """
            table = con.execute(query).pl()

            # Rename columns
            table = table.rename(
                {
                    **{
                        col: col.replace("FEATURE_MS2_", "")
                        for col in table.columns
                        if col.startswith("FEATURE_MS2_")
                    }
                }
            )

            table = table.to_pandas()

        elif self.level == "ms1":
            if not any(
                [col.startswith("FEATURE_MS1_") for col in all_precursor_column_names]
            ):
                raise click.ClickException(
                    "MS1-level feature columns are not present in precursors_features.parquet file."
                )

            # Filter columes names for FEATURE_MS1_
            feature_ms1_cols = [
                col
                for col in all_precursor_column_names
                if col.startswith("FEATURE_MS1_")
            ]
            # prepare feature ms1 columns for sql query
            feature_ms1_cols_sql = ", ".join([f"p.{col}" for col in feature_ms1_cols])

            query = f"""
                    SELECT 
                        p.RUN_ID, 
                        p.PRECURSOR_ID, 
                        p.PRECURSOR_CHARGE, 
                        p.FEATURE_ID, 
                        p.EXP_RT, 
                        p.PRECURSOR_DECOY AS DECOY, 
                        {feature_ms1_cols_sql},
                        p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
                    FROM precursors p
                    ORDER BY p.RUN_ID, p.PRECURSOR_ID ASC, p.EXP_RT ASC
                    """
            table = con.execute(query).pl()

            # Rename columns
            table = table.rename(
                {
                    **{
                        col: col.replace("FEATURE_MS1_", "")
                        for col in table.columns
                        if col.startswith("FEATURE_MS1_")
                    }
                }
            )
            table = table.to_pandas()

        elif self.level == "transition":
            if not any(
                [col.startswith("SCORE_MS2_") for col in all_precursor_column_names]
            ):
                raise click.ClickException(
                    "Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' first."
                )
            if not any(
                [
                    col.startswith("FEATURE_TRANSITION_")
                    for col in all_transition_column_names
                ]
            ):
                raise click.ClickException(
                    "Transition-level feature columns are not present in transition_features.parquet file."
                )

            # Filter columes names for FEATURE_MS1_
            feature_transition_cols = [
                col
                for col in all_transition_column_names
                if col.startswith("FEATURE_TRANSITION_VAR")
            ]
            # prepare feature ms1 columns for sql query
            feature_transition_cols_sql = ", ".join(
                [f"t.{col}" for col in feature_transition_cols]
            )

            query = f"""
            SELECT 
                t.TRANSITION_DECOY AS DECOY,
                t.FEATURE_ID AS FEATURE_ID,
                t.TRANSITION_ID AS TRANSITION_ID,
                {feature_transition_cols_sql},
                p.PRECURSOR_CHARGE AS PRECURSOR_CHARGE,
                t.TRANSITION_CHARGE AS PRODUCT_CHARGE,
                p.RUN_ID || '_' || t.FEATURE_ID || '_' || t.TRANSITION_ID AS GROUP_ID
            FROM transitions t
            INNER JOIN precursors p ON t.PRECURSOR_ID = p.PRECURSOR_ID AND t.FEATURE_ID = p.FEATURE_ID
            WHERE p.SCORE_MS2_RANK <= {self.config.runner.ipf_max_peakgroup_rank}
            AND p.SCORE_MS2_PEP <= {self.config.runner.ipf_max_peakgroup_pep}
            AND p.PRECURSOR_DECOY = 0
            AND t.FEATURE_TRANSITION_VAR_ISOTOPE_OVERLAP_SCORE <= {self.config.runner.ipf_max_transition_isotope_overlap}
            AND t.FEATURE_TRANSITION_VAR_LOG_SN_SCORE > {self.config.runner.ipf_min_transition_sn}
            ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT, t.TRANSITION_ID
            """
            table = con.execute(query).pl()

            # Rename columns
            table = table.rename(
                {
                    **{
                        col: col.replace("FEATURE_TRANSITION_", "")
                        for col in table.columns
                        if col.startswith("FEATURE_TRANSITION_")
                    }
                }
            )
            table = table.to_pandas()

        elif self.level == "alignment":
            feature_alignment_cols = [
                col for col in all_alignment_column_names if col.startswith("VAR_")
            ]
            # Prepare alignment query
            feature_alignment_cols_sql = ", ".join(
                [f"a.{col}" for col in feature_alignment_cols]
            )

            query = f"""
            SELECT 
                a.ALIGNMENT_ID,
                a.RUN_ID, 
                a.PRECURSOR_ID, 
                a.FEATURE_ID, 
                a.ALIGNED_RT, 
                a.DECOY, 
                {feature_alignment_cols_sql},
                a.RUN_ID || '_' || a.FEATURE_ID || '_' || a.PRECURSOR_ID AS GROUP_ID
            FROM alignment_features a
            ORDER BY a.RUN_ID, a.PRECURSOR_ID ASC, a.REFERENCE_RT ASC
            """
            table = con.execute(query).pl()

            table = table.to_pandas()
            #  Map DECOY to 1 and -1 to 0 and 1
            table["DECOY"] = table["DECOY"].map({1: 0, -1: 1})

        else:
            raise click.ClickException("Unspecified data level selected.")

        if self.level == "ms1ms2":
            if not any(
                [col.startswith("FEATURE_MS1_") for col in all_precursor_column_names]
            ):
                raise click.ClickException(
                    "MS1-level feature columns are not present in parquet file."
                )

            feature_ms1_cols = [
                col
                for col in all_precursor_column_names
                if col.startswith("FEATURE_MS1_VAR")
            ]
            feature_ms1_cols_sql = ", ".join([f"p.{col}" for col in feature_ms1_cols])
            query = f"""
            SELECT 
                p.FEATURE_ID, 
                {feature_ms1_cols_sql}
            FROM precursors p
            """
            ms1_table = con.execute(query).df()
            table = pd.merge(table, ms1_table, how="left", on="FEATURE_ID")

        table.columns = [col.lower() for col in table.columns]

        if ss_main_score.lower() in table.columns:
            table = table.rename(
                columns={ss_main_score.lower(): "main_" + ss_main_score.lower()}
            )
        elif ss_main_score.lower() == "swath_pretrained":
            # SWATH pretrained score is not really used anymore, so drop support for it in parquet file input workflow
            raise click.ClickException(
                "SWATH pretrained score not available for parquet files workflow"
            )
        else:
            raise click.ClickException(
                f"Main score ({ss_main_score.lower()}) column not present in data. Current columns: {table.columns}"
            )

        if self.classifier == "XGBoost" and self.level != "alignment":
            click.echo(
                "Info: Enable number of transitions & precursor / product charge scores for XGBoost-based classifier"
            )
            table = table.rename(
                columns={
                    "precursor_charge": "var_precursor_charge",
                    "product_charge": "var_product_charge",
                    "transition_count": "var_transition_count",
                }
            )

        return table

    def _read_for_ipf(self):
        # implement logic from `ipf.py`
        raise NotImplementedError

    def _read_for_context_level(self):
        # implement logic from `levels_context.py`
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

        if self.level == "ms2" or self.level == "ms1ms2":
            precursor_file = os.path.join(self.outfile, "precursors_features.parquet")

            # Process the result scores
            df = arrow_table = pa.Table.from_pandas(result.scored_tables)
            if "h_score" in df.columns:
                df = df.select(
                    [
                        "feature_id",
                        "d_score",
                        "h_score",
                        "h0_score",
                        "peak_group_rank",
                        "p_value",
                        "q_value",
                        "pep",
                    ]
                )
                df.columns = [
                    "FEATURE_ID",
                    "SCORE",
                    "HSCORE",
                    "H0SCORE",
                    "RANK",
                    "PVALUE",
                    "QVALUE",
                    "PEP",
                ]
            else:
                df = df.select(
                    [
                        "feature_id",
                        "d_score",
                        "peak_group_rank",
                        "p_value",
                        "q_value",
                        "pep",
                    ]
                )
                df.columns = ["FEATURE_ID", "SCORE", "RANK", "PVALUE", "QVALUE", "PEP"]

            # Add SCORE_MS2_ prefix to columns except FEATURE_ID
            new_columns = ["FEATURE_ID"] + [
                f"SCORE_MS2_{col}" for col in df.columns[1:]
            ]
            df = df.rename(dict(zip(df.columns, new_columns)))

            # Check and drop SCORE_MS2_ columns if they exist
            existing_cols = get_parquet_column_names(precursor_file)
            score_ms2_cols = [
                col for col in existing_cols if col.startswith("SCORE_MS2_")
            ]
            if score_ms2_cols:
                click.echo(
                    click.style(
                        "Warn: There are existing SCORE_MS2_ columns, these will be dropped.",
                        fg="yellow",
                    )
                )

            # Build the SELECT p.col list excluding old score columns
            columns_to_keep = [
                col for col in existing_cols if not col.startswith("SCORE_MS2_")
            ]
            column_list_sql = ", ".join([f"p.{col}" for col in columns_to_keep])

            # Build the SELECT s.col list for the new score columns, we don't need to include FEATURE_ID form s since it's already in p
            new_score_columns = [col for col in df.columns if col != "FEATURE_ID"]
            new_score_columns_sql = ", ".join([f"s.{col}" for col in new_score_columns])

            # Register the score table with DuckDB
            arrow_table = df.to_arrow()
            con = duckdb.connect()
            con.register("scores", arrow_table)

            # Write the new scores to the parquet file
            con.execute(
                f"""
            COPY (
                SELECT {column_list_sql}, {new_score_columns_sql}
                FROM read_parquet('{precursor_file}') p
                LEFT JOIN scores s
                ON p.FEATURE_ID = s.FEATURE_ID
            ) TO '{precursor_file}'
            (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11);
            """
            )

            click.echo("Info: %s written." % precursor_file)

        elif self.level == "ms1":
            precursor_file = os.path.join(self.outfile, "precursors_features.parquet")

            # Process the result scores
            df = pa.Table.from_pandas(result.scored_tables)
            if "h_score" in df.columns:
                df = df.select(
                    [
                        "feature_id",
                        "d_score",
                        "h_score",
                        "h0_score",
                        "peak_group_rank",
                        "p_value",
                        "q_value",
                        "pep",
                    ]
                )
                df = df.rename(
                    {
                        "feature_id": "FEATURE_ID",
                        "d_score": "SCORE",
                        "h_score": "HSCORE",
                        "h0_score": "H0SCORE",
                        "peak_group_rank": "RANK",
                        "p_value": "PVALUE",
                        "q_value": "QVALUE",
                        "pep": "PEP",
                    }
                )
            else:
                df = df.select(
                    [
                        "feature_id",
                        "d_score",
                        "peak_group_rank",
                        "p_value",
                        "q_value",
                        "pep",
                    ]
                )
                df = df.rename(
                    {
                        "feature_id": "FEATURE_ID",
                        "d_score": "SCORE",
                        "peak_group_rank": "RANK",
                        "p_value": "PVALUE",
                        "q_value": "QVALUE",
                        "pep": "PEP",
                    }
                )

            # Add SCORE_MS1_ prefix to columns except FEATURE_ID
            new_columns = {
                col: f"SCORE_MS1_{col}" if col != "FEATURE_ID" else col
                for col in df.columns
            }
            df = df.rename(new_columns)

            # Check and drop SCORE_MS1_ columns if they exist
            existing_cols = get_parquet_column_names(precursor_file)
            score_ms1_cols = [
                col for col in existing_cols if col.startswith("SCORE_MS1_")
            ]
            if score_ms1_cols:
                click.echo(
                    click.style(
                        "Warn: There are existing SCORE_MS1_ columns, these will be dropped.",
                        fg="yellow",
                    )
                )

            # Build the SELECT p.col list excluding old score columns
            columns_to_keep = [
                col for col in existing_cols if not col.startswith("SCORE_MS1_")
            ]
            column_list_sql = ", ".join([f"p.{col}" for col in columns_to_keep])

            # Build the SELECT s.col list for the new score columns, we don't need to include FEATURE_ID form s since it's already in p
            new_score_columns = [col for col in df.columns if col != "FEATURE_ID"]
            new_score_columns_sql = ", ".join([f"s.{col}" for col in new_score_columns])

            # Register the score table with DuckDB
            arrow_table = df.to_arrow()
            con = duckdb.connect()
            con.register("scores", arrow_table)

            # Write the new scores to the parquet file
            con.execute(
                f"""
            COPY (
                SELECT {column_list_sql}, {new_score_columns_sql}
                FROM read_parquet('{precursor_file}') p
                LEFT JOIN scores s
                ON p.FEATURE_ID = s.FEATURE_ID
            ) TO '{precursor_file}'
            (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11);
            """
            )
            click.echo("Info: %s written." % precursor_file)

        elif self.level == "transition":
            transition_file = os.path.join(self.outfile, "transition_features.parquet")

            # Select and rename columns
            df = df[
                [
                    "feature_id",
                    "transition_id",
                    "d_score",
                    "peak_group_rank",
                    "p_value",
                    "q_value",
                    "pep",
                ]
            ].rename(
                columns={
                    "feature_id": "FEATURE_ID",
                    "transition_id": "TRANSITION_ID",
                    "d_score": "SCORE",
                    "peak_group_rank": "RANK",
                    "p_value": "PVALUE",
                    "q_value": "QVALUE",
                    "pep": "PEP",
                }
            )

            # Add SCORE_TRANSITION_ prefix to all except FEATURE_ID and TRANSITION_ID
            df = df.rename(
                columns={
                    col: f"SCORE_TRANSITION_{col}"
                    for col in df.columns
                    if col not in ["FEATURE_ID", "TRANSITION_ID"]
                }
            )

            # Check for existing SCORE_TRANSITION_ columns
            existing_cols = get_parquet_column_names(transition_file)
            score_transition_cols = [
                col for col in existing_cols if col.startswith("SCORE_TRANSITION_")
            ]
            if score_transition_cols:
                click.echo(
                    click.style(
                        "Warn: There are existing SCORE_TRANSITION_ columns, these will be dropped.",
                        fg="yellow",
                    )
                )

            # Build the SELECT p.col list excluding old score columns
            columns_to_keep = [
                col for col in existing_cols if not col.startswith("SCORE_TRANSITION_")
            ]
            column_list_sql = ", ".join([f"t.{col}" for col in columns_to_keep])
            # Build the SELECT s.col list for the new score columns, we don't need to include FEATURE_ID form s since it's already in p
            new_score_columns = [
                col for col in df.columns if col not in ["FEATURE_ID", "TRANSITION_ID"]
            ]
            new_score_columns_sql = ", ".join([f"s.{col}" for col in new_score_columns])

            # Register the score table with DuckDB
            arrow_table = pa.Table.from_pandas(df)
            con = duckdb.connect()
            con.register("scores", arrow_table)

            # Write the new scores to the parquet file
            con.execute(
                f"""
            COPY (
                SELECT {column_list_sql}, {new_score_columns_sql}
                FROM read_parquet('{transition_file}') t
                LEFT JOIN scores s
                ON t.FEATURE_ID = s.FEATURE_ID and t.TRANSITION_ID = s.TRANSITION_ID
            ) TO '{transition_file}'
            (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11);
            """
            )

            click.echo("Info: %s written." % transition_file)

        elif self.level == "alignment":
            alignment_file = os.path.join(self.outfile, "feature_alignment.parquet")

            # Filter out decoy rows
            df = df[df["decoy"] == 0]

            # Select and rename columns
            df = df[
                [
                    "feature_id",
                    "d_score",
                    "peak_group_rank",
                    "p_value",
                    "q_value",
                    "pep",
                ]
            ].rename(
                columns={
                    "feature_id": "FEATURE_ID",
                    "d_score": "SCORE",
                    "peak_group_rank": "RANK",
                    "p_value": "PVALUE",
                    "q_value": "QVALUE",
                    "pep": "PEP",
                }
            )

            # Add SCORE_ALIGNMENT_ prefix to all columns except FEATURE_ID
            df = df.rename(
                columns={
                    col: f"SCORE_ALIGNMENT_{col}"
                    for col in df.columns
                    if col != "FEATURE_ID"
                }
            )

            # Check for existing SCORE_ALIGNMENT_ columns
            existing_cols = get_parquet_column_names(alignment_file)
            score_alignment_cols = [
                col for col in existing_cols if col.startswith("SCORE_ALIGNMENT_")
            ]
            if score_alignment_cols:
                click.echo(
                    click.style(
                        "Warn: There are existing SCORE_ALIGNMENT_ columns, these will be dropped.",
                        fg="yellow",
                    )
                )

            # Build the SELECT p.col list excluding old score columns
            columns_to_keep = [
                col for col in existing_cols if not col.startswith("SCORE_ALIGNMENT_")
            ]
            column_list_sql = ", ".join([f"a.{col}" for col in columns_to_keep])

            # Build the SELECT s.col list for the new score columns, we don't need to include FEATURE_ID form s since it's already in p
            new_score_columns = [col for col in df.columns if col != "FEATURE_ID"]
            new_score_columns_sql = ", ".join([f"s.{col}" for col in new_score_columns])

            # Register the score table with DuckDB
            arrow_table = df.to_arrow()
            con = duckdb.connect()
            con.register("scores", arrow_table)

            # Write the new scores to the parquet file
            con.execute(
                f"""
            COPY (
                SELECT {column_list_sql}, {new_score_columns_sql}
                FROM read_parquet('{alignment_file}') a
                LEFT JOIN scores s
                ON a.FEATURE_ID = s.FEATURE_ID
            ) TO '{alignment_file}'
            (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11);
            """
            )

            click.echo("Info: %s written." % alignment_file)

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[
                (result.scored_tables.peak_group_rank == 1)
                & (result.scored_tables.decoy == 0)
            ]["p_value"].values
            top_targets = result.scored_tables.loc[
                (result.scored_tables.peak_group_rank == 1)
                & (result.scored_tables.decoy == 0)
            ]["d_score"].values
            top_decoys = result.scored_tables.loc[
                (result.scored_tables.peak_group_rank == 1)
                & (result.scored_tables.decoy == 1)
            ]["d_score"].values

            save_report(
                os.path.join(
                    self.config.runner.prefix + "_" + self.level + "_report.pdf"
                ),
                self.outfile,
                top_decoys,
                top_targets,
                cutoffs,
                svalues,
                qvalues,
                pvalues,
                pi0,
                self.config.runner.color_palette,
            )
            click.echo(
                "Info: %s written."
                % os.path.join(
                    self.config.runner.prefix + "_" + self.level + "_report.pdf"
                )
            )

    def _save_ipf_results(self, result):
        # extract logic from ipf.py
        raise NotImplementedError

    def _save_context_level_results(self, result):
        # extract logic from levels_context.py
        raise NotImplementedError
