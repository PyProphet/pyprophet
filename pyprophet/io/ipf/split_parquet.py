import os
import glob
from shutil import copyfile
from typing import Literal
import pandas as pd
import pyarrow as pa
import duckdb
import click
from loguru import logger

from ..util import get_parquet_column_names
from .._base import BaseSplitParquetReader, BaseSplitParquetWriter
from ..._config import IPFIOConfig


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

    def __init__(self, config: IPFIOConfig):
        super().__init__(config)

    def read(
        self, level: Literal["peakgroup_precursor", "transition", "alignment"]
    ) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            self._init_duckdb_views(con)

            if level == "peakgroup_precursor":
                return self._read_pyp_peakgroup_precursor(con)
            elif level == "transition":
                return self._read_pyp_transition(con)
            elif level == "alignment":
                return self._fetch_alignment_features(con)
            else:
                raise click.ClickException(f"Unsupported level: {level}")
        finally:
            con.close()

    def _read_pyp_peakgroup_precursor(self, con) -> pd.DataFrame:
        cfg = self.config
        ipf_ms1 = cfg.ipf_ms1_scoring
        ipf_ms2 = cfg.ipf_ms2_scoring
        pep_threshold = cfg.ipf_max_peakgroup_pep

        logger.info("Reading precursor-level data ...")

        if cfg.file_type == "parquet_split_multi":
            precursor_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "precursors_features.parquet")
            )
            transition_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "transition_features.parquet")
            )
        else:
            precursor_files = [os.path.join(self.infile, "precursors_features.parquet")]
            transition_files = [
                os.path.join(self.infile, "transition_features.parquet")
            ]

        all_precursor_cols = get_parquet_column_names(precursor_files[0])
        all_transition_cols = get_parquet_column_names(transition_files[0])

        # con.execute(
        #     f"CREATE VIEW precursors AS SELECT * FROM read_parquet({precursor_files})"
        # )
        # con.execute(
        #     f"CREATE VIEW transitions AS SELECT * FROM read_parquet({transition_files})"
        # )

        if not ipf_ms1 and ipf_ms2:
            if not any(
                c.startswith("SCORE_MS2_") for c in all_precursor_cols
            ) or not any(
                c.startswith("SCORE_TRANSITION_") for c in all_transition_cols
            ):
                raise click.ClickException("Apply MS2 + transition scoring before IPF.")
            query = f"""
            SELECT p.FEATURE_ID, p.SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                NULL AS MS1_PRECURSOR_PEP, t.SCORE_TRANSITION_PEP AS MS2_PRECURSOR_PEP
            FROM precursors p
            INNER JOIN (
                SELECT FEATURE_ID, SCORE_TRANSITION_PEP
                FROM transition
                WHERE TRANSITION_TYPE = '' AND TRANSITION_DECOY = 0
            ) t ON p.FEATURE_ID = t.FEATURE_ID
            WHERE p.PRECURSOR_DECOY = 0 AND p.SCORE_MS2_PEP < {pep_threshold};
            """

        elif ipf_ms1 and not ipf_ms2:
            if not any(
                c.startswith("SCORE_MS1_") for c in all_precursor_cols
            ) or not any(c.startswith("SCORE_MS2_") for c in all_precursor_cols):
                raise click.ClickException("Apply MS1 + MS2 scoring before IPF.")
            query = f"""
            SELECT p.FEATURE_ID, p.SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                p.SCORE_MS1_PEP AS MS1_PRECURSOR_PEP, NULL AS MS2_PRECURSOR_PEP
            FROM precursors p
            WHERE p.PRECURSOR_DECOY = 0 AND p.SCORE_MS2_PEP < {pep_threshold};
            """

        elif ipf_ms1 and ipf_ms2:
            if not all(
                [
                    any(c.startswith("SCORE_MS1_") for c in all_precursor_cols),
                    any(c.startswith("SCORE_MS2_") for c in all_precursor_cols),
                    any(c.startswith("SCORE_TRANSITION_") for c in all_transition_cols),
                ]
            ):
                raise click.ClickException(
                    "Apply MS1 + MS2 + transition scoring before IPF."
                )
            query = f"""
            SELECT p.FEATURE_ID, p.SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                p.SCORE_MS1_PEP AS MS1_PRECURSOR_PEP, t.SCORE_TRANSITION_PEP AS MS2_PRECURSOR_PEP
            FROM precursors p
            INNER JOIN (
                SELECT FEATURE_ID, SCORE_TRANSITION_PEP
                FROM transition
                WHERE TRANSITION_TYPE = '' AND TRANSITION_DECOY = 0
            ) t ON p.FEATURE_ID = t.FEATURE_ID
            WHERE p.PRECURSOR_DECOY = 0 AND p.SCORE_MS2_PEP < {pep_threshold};
            """

        else:
            if not any(
                c.startswith("SCORE_MS2_") for c in all_precursor_cols
            ) or not any(
                c.startswith("SCORE_TRANSITION_") for c in all_transition_cols
            ):
                raise click.ClickException("Apply MS2 + transition scoring before IPF.")
            query = f"""
            SELECT p.FEATURE_ID, p.SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                NULL AS MS1_PRECURSOR_PEP, NULL AS MS2_PRECURSOR_PEP
            FROM precursors p
            WHERE p.PRECURSOR_DECOY = 0 AND p.SCORE_MS2_PEP < {pep_threshold};
            """

        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]
        return df

    def _read_pyp_transition(self, con) -> pd.DataFrame:
        cfg = self.config
        ipf_h0 = cfg.ipf_h0
        pep_threshold = cfg.ipf_max_transition_pep

        logger.info("Reading peptidoform-level data ...")

        # Resolve transition file paths
        if cfg.file_type == "parquet_split_multi":
            transition_files = glob.glob(
                os.path.join(self.infile, "*.oswpq", "transition_features.parquet")
            )
        else:
            transition_files = [
                os.path.join(self.infile, "transition_features.parquet")
            ]

        if not transition_files:
            raise click.ClickException("No transition_features.parquet files found.")

        # Use first file for column check
        all_transition_cols = get_parquet_column_names(transition_files[0])

        if "IPF_PEPTIDE_ID" not in all_transition_cols:
            raise click.ClickException(
                "IPF_PEPTIDE_ID column is required in transition features."
            )

        # Load all transition files into DuckDB view
        con.execute(
            f"CREATE VIEW transitions AS SELECT * FROM read_parquet({transition_files})"
        )

        # Evidence table: transition-level PEPs
        query = f"""
        SELECT t.FEATURE_ID, t.TRANSITION_ID, t.SCORE_TRANSITION_PEP AS PEP
        FROM transition t
        WHERE t.TRANSITION_TYPE != ''
        AND t.TRANSITION_DECOY = 0
        AND t.SCORE_TRANSITION_SCORE IS NOT NULL
        AND t.SCORE_TRANSITION_PEP < {pep_threshold}
        """
        evidence = con.execute(query).df().rename(columns=str.lower)

        # Bitmask table: transition-peptidoform presence
        query = """
        SELECT DISTINCT t.TRANSITION_ID, t.IPF_PEPTIDE_ID AS PEPTIDE_ID, 1 AS BMASK
        FROM transition t
        WHERE t.TRANSITION_TYPE != ''
        AND t.TRANSITION_DECOY = 0
        AND t.SCORE_TRANSITION_SCORE IS NOT NULL
        AND t.IPF_PEPTIDE_ID IS NOT NULL
        """
        bitmask = con.execute(query).df().rename(columns=str.lower)

        # Peptidoform count per feature
        query = """
        SELECT t.FEATURE_ID, COUNT(DISTINCT t.IPF_PEPTIDE_ID) AS NUM_PEPTIDOFORMS
        FROM transition t
        WHERE t.TRANSITION_TYPE != ''
        AND t.TRANSITION_DECOY = 0
        AND t.SCORE_TRANSITION_SCORE IS NOT NULL
        AND t.IPF_PEPTIDE_ID IS NOT NULL
        GROUP BY t.FEATURE_ID
        ORDER BY t.FEATURE_ID
        """
        num_peptidoforms = con.execute(query).df().rename(columns=str.lower)

        # Peptidoform mapping: (feature_id, peptide_id)
        query = """
        SELECT DISTINCT t.FEATURE_ID, t.IPF_PEPTIDE_ID AS PEPTIDE_ID
        FROM transition t
        WHERE t.TRANSITION_TYPE != ''
        AND t.TRANSITION_DECOY = 0
        AND t.SCORE_TRANSITION_SCORE IS NOT NULL
        AND t.IPF_PEPTIDE_ID IS NOT NULL
        ORDER BY t.FEATURE_ID
        """
        peptidoforms = con.execute(query).df().rename(columns=str.lower)

        # Add h0 (decoy) peptide_id = -1 if enabled
        if ipf_h0:
            h0_df = pd.DataFrame(
                {"feature_id": peptidoforms["feature_id"].unique(), "peptide_id": -1}
            )
            peptidoforms = pd.concat([peptidoforms, h0_df], ignore_index=True)

        # Merge all parts into final dataframe
        trans_pf = pd.merge(evidence, peptidoforms, how="outer", on="feature_id")
        trans_pf_bm = pd.merge(
            trans_pf, bitmask, how="left", on=["transition_id", "peptide_id"]
        ).fillna(0)
        result = pd.merge(trans_pf_bm, num_peptidoforms, how="inner", on="feature_id")

        logger.info(f"Loaded {len(result)} transition-peptidoform entries")
        return result

    def _fetch_alignment_features(self, con) -> pd.DataFrame:
        logger.info("Reading Across Run Feature Alignment Mapping ...")

        pep_threshold = self.config.ipf_max_alignment_pep
        alignment_file = os.path.join(self.infile, "feature_alignment.parquet")

        if not os.path.exists(alignment_file):
            raise click.ClickException(f"Alignment file not found: {alignment_file}")

        # Read alignment file into DuckDB
        con.execute(
            f"""
            CREATE VIEW alignment_features AS 
            SELECT * FROM read_parquet('{alignment_file}')
        """
        )

        query = f"""
            SELECT
                DENSE_RANK() OVER (ORDER BY a.PRECURSOR_ID, a.ALIGNMENT_ID) AS ALIGNMENT_GROUP_ID,
                a.FEATURE_ID AS FEATURE_ID
            FROM alignment_features AS a
            WHERE DECOY = 1
            AND a.SCORE_ALIGNMENT_PEP < {pep_threshold}
            ORDER BY ALIGNMENT_GROUP_ID
        """

        df = con.execute(query).df()
        df.columns = [col.lower() for col in df.columns]

        logger.info(f"Loaded {len(df)} aligned feature group mappings.")
        logger.debug(f"Unique alignment groups: {df['alignment_group_id'].nunique()}")

        return df


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

    def __init__(self, config: IPFIOConfig):
        super().__init__(config)

    def save_results(self, result):
        df = result

        # Rename score columns
        df = df.rename(
            columns={
                "PRECURSOR_PEAKGROUP_PEP": "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP",
                "QVALUE": "SCORE_IPF_QVALUE",
                "PEP": "SCORE_IPF_PEP",
            }
        )

        # Identify columns to merge
        score_cols = [
            "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP",
            "SCORE_IPF_QVALUE",
            "SCORE_IPF_PEP",
        ]
        join_cols = ["FEATURE_ID", "PEPTIDE_ID"]

        # Make sure the input dataframe has these columns
        if not all(col in df.columns for col in join_cols + score_cols):
            raise click.ClickException("Missing required columns in result dataframe.")

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

            # Read FEATURE_IDs from current file
            try:
                con = duckdb.connect()
                feature_ids = con.execute(
                    f"SELECT DISTINCT FEATURE_ID FROM read_parquet('{file_path}')"
                ).fetchall()
                con.close()
            except Exception as e:
                logger.error(f"Error reading FEATURE_IDs from {file_path}: {e}")
                continue

            feature_ids = set(f[0] for f in feature_ids)
            subset = df[df["FEATURE_ID"].isin(feature_ids)]

            if subset.empty:
                logger.warning(
                    f"No matching FEATURE_IDs found for {run_dir}, skipping."
                )
                continue

            # Identify columns to keep from original parquet file
            existing_cols = get_parquet_column_names(file_path)
            score_ipf_cols = [
                col for col in existing_cols if col.startswith("SCORE_IPF")
            ]
            if score_ipf_cols:
                logger.warning(
                    "Warn: There are existing SCORE_IPF_ columns, these will be dropped."
                )
            existing_cols = [
                col for col in existing_cols if not col.startswith("SCORE_IPF_")
            ]
            select_old = ", ".join([f"p.{col}" for col in existing_cols])
            new_score_sql = ", ".join([f"s.{col}" for col in score_cols])

            con = duckdb.connect()
            con.register("scores", pa.Table.from_pandas(subset))

            # Validate input row entry count and joined entry count remain the same
            self._validate_row_count_after_join(
                con,
                file_path,
                "p.FEATURE_ID, p.IPF_PEPTIDE_ID",
                "p.FEATURE_ID = s.FEATURE_ID AND p.IPF_PEPTIDE_ID = s.PEPTIDE_ID",
                "p",
            )

            con.execute(
                f"""
                COPY (
                    SELECT {select_old}, {new_score_sql}
                    FROM read_parquet('{file_path}') p
                    LEFT JOIN scores s
                    ON p.FEATURE_ID = s.FEATURE_ID AND p.IPF_PEPTIDE_ID = s.PEPTIDE_ID
                ) TO '{file_path}'
                (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11)
                """
            )

            logger.debug(
                f"After appendings scores, {file_path} has {self._get_parquet_row_count(con, file_path)} entries"
            )

            con.close()
            logger.success(f"Updated: {file_path}")
