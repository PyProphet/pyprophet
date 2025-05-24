import os
from typing import Literal
from shutil import copyfile
import pandas as pd
import pyarrow as pa
import duckdb
import click
from loguru import logger
from ..util import get_parquet_column_names
from .._base import BaseReader, BaseWriter
from ..._config import IPFIOConfig


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

    def __init__(self, config: IPFIOConfig):
        super().__init__(config)

        if config.propagate_signal_across_runs:
            # We make the assumption that the alignment file is in the same directory as the input file
            self.alignment_file = os.path.join(
                os.path.dirname(self.infile), "feature_alignment.parquet"
            )
            if not os.path.exists(self.alignment_file):
                raise click.ClickException(
                    f"To use the --propagate-signal-across-runs option, "
                    f"the alignment file {self.alignment_file} must exist."
                )

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

    def _init_duckdb_views(self, con):
        con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{self.infile}')")

        if self.config.propagate_signal_across_runs:
            con.execute(
                f"CREATE VIEW alignment AS SELECT * FROM read_parquet('{self.alignment_file}')"
            )

    def _get_columns_by_prefix(self, parquet_file, prefix):
        cols = get_parquet_column_names(parquet_file)
        return [c for c in cols if c.startswith(prefix)]

    def _read_pyp_peakgroup_precursor(self, con) -> pd.DataFrame:
        cfg = self.config
        ipf_ms1 = cfg.ipf_ms1_scoring
        ipf_ms2 = cfg.ipf_ms2_scoring
        pep_threshold = cfg.ipf_max_peakgroup_pep

        logger.info("Reading precursor-level data ...")

        all_cols = get_parquet_column_names(self.infile)

        if not ipf_ms1 and ipf_ms2:
            if not any(c.startswith("SCORE_MS2_") for c in all_cols) or not any(
                c.startswith("SCORE_TRANSITION_") for c in all_cols
            ):
                raise click.ClickException("Apply MS2 + transition scoring before IPF.")
            query = f"""
            SELECT FEATURE_ID,
                SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                NULL AS MS1_PRECURSOR_PEP,
                SCORE_TRANSITION_PEP AS MS2_PRECURSOR_PEP
            FROM data
            WHERE PRECURSOR_DECOY = 0
            AND TRANSITION_TYPE = ''
            AND TRANSITION_DECOY = 0
            AND SCORE_MS2_PEP < {pep_threshold}
            """

        elif ipf_ms1 and not ipf_ms2:
            if not any(c.startswith("SCORE_MS1_") for c in all_cols) or not any(
                c.startswith("SCORE_MS2_") for c in all_cols
            ):
                raise click.ClickException("Apply MS1 + MS2 scoring before IPF.")
            query = f"""
            SELECT FEATURE_ID,
                SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                SCORE_MS1_PEP AS MS1_PRECURSOR_PEP,
                NULL AS MS2_PRECURSOR_PEP
            FROM data
            WHERE PRECURSOR_DECOY = 0
            AND SCORE_MS2_PEP < {pep_threshold}
            """

        elif ipf_ms1 and ipf_ms2:
            if not all(
                [
                    any(c.startswith("SCORE_MS1_") for c in all_cols),
                    any(c.startswith("SCORE_MS2_") for c in all_cols),
                    any(c.startswith("SCORE_TRANSITION_") for c in all_cols),
                ]
            ):
                raise click.ClickException(
                    "Apply MS1 + MS2 + transition scoring before IPF."
                )
            query = f"""
            SELECT FEATURE_ID,
                SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                SCORE_MS1_PEP AS MS1_PRECURSOR_PEP,
                SCORE_TRANSITION_PEP AS MS2_PRECURSOR_PEP
            FROM data
            WHERE PRECURSOR_DECOY = 0
            AND TRANSITION_TYPE = ''
            AND TRANSITION_DECOY = 0
            AND SCORE_MS2_PEP < {pep_threshold}
            """

        else:
            if not any(c.startswith("SCORE_MS2_") for c in all_cols) or not any(
                c.startswith("SCORE_TRANSITION_") for c in all_cols
            ):
                raise click.ClickException("Apply MS2 + transition scoring before IPF.")
            query = f"""
            SELECT FEATURE_ID,
                SCORE_MS2_PEP AS MS2_PEAKGROUP_PEP,
                NULL AS MS1_PRECURSOR_PEP,
                NULL AS MS2_PRECURSOR_PEP
            FROM data
            WHERE PRECURSOR_DECOY = 0
            AND SCORE_MS2_PEP < {pep_threshold}
            """

        df = con.execute(query).df()
        return df.rename(columns=str.lower)

    def _read_pyp_transition(self, con) -> pd.DataFrame:
        cfg = self.config
        ipf_h0 = cfg.ipf_h0
        pep_threshold = cfg.ipf_max_transition_pep

        logger.info("Reading peptidoform-level data ...")

        # Check required columns exist
        all_cols = get_parquet_column_names(self.infile)
        if "IPF_PEPTIDE_ID" not in all_cols:
            raise click.ClickException(
                "IPF_PEPTIDE_ID column is required in transition features."
            )

        # Use DuckDB view `data` created in _init_duckdb_views()

        # Evidence table: transition-level PEPs
        query = f"""
        SELECT FEATURE_ID, TRANSITION_ID, SCORE_TRANSITION_PEP AS PEP
        FROM data
        WHERE TRANSITION_TYPE != ''
        AND TRANSITION_DECOY = 0
        AND SCORE_TRANSITION_PEP < {pep_threshold}
        """
        evidence = con.execute(query).df().rename(columns=str.lower)

        # Bitmask table: transition-peptidoform presence
        query = """
        SELECT DISTINCT TRANSITION_ID, IPF_PEPTIDE_ID AS PEPTIDE_ID, 1 AS BMASK
        FROM data
        WHERE TRANSITION_TYPE != ''
        AND TRANSITION_DECOY = 0
        AND IPF_PEPTIDE_ID IS NOT NULL
        """
        bitmask = con.execute(query).df().rename(columns=str.lower)

        # Peptidoform count per feature
        query = """
        SELECT FEATURE_ID, COUNT(DISTINCT IPF_PEPTIDE_ID) AS NUM_PEPTIDOFORMS
        FROM data
        WHERE TRANSITION_TYPE != ''
        AND TRANSITION_DECOY = 0
        AND IPF_PEPTIDE_ID IS NOT NULL
        GROUP BY FEATURE_ID
        ORDER BY FEATURE_ID
        """
        num_peptidoforms = con.execute(query).df().rename(columns=str.lower)

        # Peptidoform mapping: (feature_id, peptide_id)
        query = """
        SELECT DISTINCT FEATURE_ID, IPF_PEPTIDE_ID AS PEPTIDE_ID
        FROM data
        WHERE TRANSITION_TYPE != ''
        AND TRANSITION_DECOY = 0
        AND IPF_PEPTIDE_ID IS NOT NULL
        ORDER BY FEATURE_ID
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

        # Read alignment file into DuckDB
        con.execute(
            f"""
            CREATE VIEW alignment_features AS 
            SELECT * FROM read_parquet('{self.alignment_file}')
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

    def __init__(self, config: IPFIOConfig):
        super().__init__(config)

    def save_results(self, result):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        target_file = self.outfile

        # Rename score columnes with SCORE_IPF_
        result = result.rename(
            columns={
                "PRECURSOR_PEAKGROUP_PEP": "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP",
                "QVALUE": "SCORE_IPF_QVALUE",
                "PEP": "SCORE_IPF_PEP",
            }
        )

        # Register the score table with DuckDB
        con = duckdb.connect()
        con.register("scores", pa.Table.from_pandas(result))

        # Check and drop SCORE_IPF_ columns if they exist
        existing_cols = get_parquet_column_names(target_file)
        score_ipf_cols = [col for col in existing_cols if col.startswith("SCORE_IPF")]
        if score_ipf_cols:
            logger.warning(
                "Warn: There are existing SCORE_IPF_ columns, these will be dropped."
            )

        # Build the SELECT p.col list excluding old score columns
        columns_to_keep = [
            col for col in existing_cols if not col.startswith("SCORE_IPF_")
        ]
        column_list_sql = ", ".join([f"p.{col}" for col in columns_to_keep])

        # Validate input row entry count and joined entry count remain the same
        self._validate_row_count_after_join(
            con,
            target_file,
            "p.FEATURE_ID, p.IPF_PEPTIDE_ID",
            "ON p.FEATURE_ID = s.FEATURE_ID AND p.IPF_PEPTIDE_ID = s.PEPTIDE_ID",
            "p",
        )

        # Write the new scores to the parquet file
        con.execute(
            f"""
            COPY (
                SELECT 
                    {column_list_sql},
                    s.SCORE_IPF_PRECURSOR_PEAKGROUP_PEP,
                    s.SCORE_IPF_QVALUE,
                    s.SCORE_IPF_PEP
                FROM read_parquet('{target_file}') p
                LEFT JOIN scores s
                    ON p.FEATURE_ID = s.FEATURE_ID 
                AND p.IPF_PEPTIDE_ID = s.PEPTIDE_ID
            ) TO '{target_file}'
            (FORMAT 'parquet', COMPRESSION 'ZSTD', COMPRESSION_LEVEL 11);
        """
        )

        logger.debug(
            f"After appendings scores, {target_file} has {self._get_parquet_row_count(con, target_file)} entries"
        )

        con.close()

        logger.success(f"{target_file} written.")
