import os
import pickle
from shutil import copyfile
import sqlite3
from typing import Literal, Tuple
import re
import duckdb
import pandas as pd
import numpy as np
import click
from loguru import logger
from ..util import (
    check_sqlite_table,
    check_duckdb_table,
    unimod_to_codename,
    write_scores_sql_command,
    load_sqlite_scanner,
    get_table_columns,
    get_table_columns_with_types,
)
from .._base import BaseOSWReader, BaseOSWWriter
from ..._config import ExportIOConfig


class OSWReader(BaseOSWReader):
    """
    Class for reading and processing data from an OpenSWATH workflow OSW-sqlite based file.
    Extended to support exporting functionality.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        """
        Read data from the OpenSWATH workflow OSW-sqlite based file.
        """
        con = sqlite3.connect(self.infile)
        return self._read_sqlite(con)

    def _create_indexes(self):
        """
        Create necessary indexes for export queries.
        """
        try:
            sqlite_con = sqlite3.connect(self.infile)

            index_statements = [
                "CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);",
                "CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);",
                "CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);",
                "CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);",
                "CREATE INDEX IF NOT EXISTS idx_run_run_id ON RUN (ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_run_id ON FEATURE (RUN_ID);",
                "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);",
            ]

            # Add conditional indexes based on tables present
            if check_sqlite_table(sqlite_con, "FEATURE_MS1"):
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);"
                )
            if check_sqlite_table(sqlite_con, "FEATURE_MS2"):
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);"
                )
            if check_sqlite_table(sqlite_con, "SCORE_MS1"):
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_score_ms1_feature_id ON SCORE_MS1 (FEATURE_ID);"
                )
            if check_sqlite_table(sqlite_con, "SCORE_MS2"):
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);"
                )
            if check_sqlite_table(sqlite_con, "SCORE_IPF"):
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_score_ipf_feature_id ON SCORE_IPF (FEATURE_ID);"
                )
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_score_ipf_peptide_id ON SCORE_IPF (PEPTIDE_ID);"
                )
            if check_sqlite_table(sqlite_con, "SCORE_TRANSITION"):
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);"
                )
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_transition_transition_id ON TRANSITION (ID);"
                )
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id_feature_id ON FEATURE_TRANSITION (TRANSITION_ID, FEATURE_ID);"
                )
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id_feature_id ON SCORE_TRANSITION (TRANSITION_ID, FEATURE_ID);"
                )
                index_statements.append(
                    "CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID);"
                )

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

    def _read_duckdb(self, con):
        cfg = self.config
        return self._read_sqlite(con)  # We will use SQLite as the main reader for now

    def _read_sqlite(self, con):
        """Main entry point for reading SQLite data, delegates to specific methods."""
        cfg = self.config

        if self._is_unscored_file(con):
            logger.info("Reading unscored data from Parquet file.")
            return self._read_unscored_data(con)

        ipf_present = self._check_ipf_presence(con, cfg)

        if ipf_present and cfg.ipf == "peptidoform":
            logger.info("Reading peptidoform IPF data from Parquet file.")
            data = self._read_peptidoform_data(con, cfg)
        elif ipf_present and cfg.ipf == "augmented":
            logger.info("Reading augmented data with IPF from Parquet file.")
            data = self._read_augmented_data(con, cfg)
        else:
            logger.info("Reading standard OpenSWATH data from Parquet file.")
            data = self._read_standard_data(con, cfg)

        # Apply common augmentations to all scored data types
        return self._augment_data(data, con, cfg)

    def _is_unscored_file(self, con):
        """Check if the file is unscored (no score tables present)."""
        tables = [
            "SCORE_MS1",
            "SCORE_MS2",
            "SCORE_IPF",
            "SCORE_PEPTIDE",
            "SCORE_PROTEIN",
        ]
        return all(not check_sqlite_table(con, table) for table in tables)

    def _check_ipf_presence(self, con, cfg):
        """Check if IPF data is present and should be used."""
        return cfg.ipf != "disable" and check_sqlite_table(con, "SCORE_IPF")

    def _read_unscored_data(self, con):
        """Read data from unscored files."""
        score_sql = self._build_score_sql(con)

        query = f"""
            SELECT
                RUN.ID AS id_run,
                PEPTIDE.ID AS id_peptide,
                PRECURSOR.ID AS transition_group_id,
                PRECURSOR.DECOY AS decoy,
                RUN.ID AS run_id,
                RUN.FILENAME AS filename,
                FEATURE.EXP_RT AS RT,
                FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
                FEATURE.DELTA_RT AS delta_rt,
                PRECURSOR.LIBRARY_RT AS assay_RT,
                FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_RT,
                FEATURE.ID AS id,
                PRECURSOR.CHARGE AS Charge,
                PRECURSOR.PRECURSOR_MZ AS mz,
                FEATURE_MS2.AREA_INTENSITY AS Intensity,
                FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
                FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
                FEATURE.LEFT_WIDTH AS leftWidth,
                FEATURE.RIGHT_WIDTH AS rightWidth
                {score_sql}
            FROM PRECURSOR
            INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
            LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
            LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
            ORDER BY transition_group_id
        """
        return pd.read_sql_query(query, con)

    def _build_score_sql(self, con):
        """Build SQL fragment for score columns in unscored files."""
        score_sql = ""
        if check_sqlite_table(con, "FEATURE_MS1"):
            score_sql = write_scores_sql_command(
                con, score_sql, "FEATURE_MS1", "var_ms1_"
            )
        if check_sqlite_table(con, "FEATURE_MS2"):
            score_sql = write_scores_sql_command(
                con, score_sql, "FEATURE_MS2", "var_ms2_"
            )

        if score_sql:
            return ", " + score_sql[:-2]  # Remove last comma and space
        return ""

    def _read_peptidoform_data(self, con, cfg):
        """Read data with peptidoform IPF information."""
        score_ms1_pep, link_ms1 = self._get_ms1_score_info(con)

        query = f"""
            SELECT RUN.ID AS id_run,
                  PEPTIDE.ID AS id_peptide,
                  PEPTIDE_IPF.MODIFIED_SEQUENCE || '_' || PRECURSOR.ID AS transition_group_id,
                  PRECURSOR.DECOY AS decoy,
                  RUN.ID AS run_id,
                  RUN.FILENAME AS filename,
                  FEATURE.EXP_RT AS RT,
                  FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
                  FEATURE.DELTA_RT AS delta_rt,
                  FEATURE.NORM_RT AS iRT,
                  PRECURSOR.LIBRARY_RT AS assay_iRT,
                  FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT,
                  FEATURE.ID AS id,
                  PEPTIDE_IPF.UNMODIFIED_SEQUENCE AS Sequence,
                  PEPTIDE_IPF.MODIFIED_SEQUENCE AS FullPeptideName,
                  PRECURSOR.CHARGE AS Charge,
                  PRECURSOR.PRECURSOR_MZ AS mz,
                  FEATURE_MS2.AREA_INTENSITY AS Intensity,
                  FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
                  FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
                  FEATURE.LEFT_WIDTH AS leftWidth,
                  FEATURE.RIGHT_WIDTH AS rightWidth,
                  {score_ms1_pep} AS ms1_pep,
                  SCORE_MS2.PEP AS ms2_pep,
                  SCORE_IPF.PRECURSOR_PEAKGROUP_PEP AS precursor_pep,
                  SCORE_IPF.PEP AS ipf_pep,
                  SCORE_MS2.RANK AS peak_group_rank,
                  SCORE_MS2.SCORE AS d_score,
                  SCORE_MS2.QVALUE AS ms2_m_score,
                  SCORE_IPF.QVALUE AS m_score
            FROM PRECURSOR
            INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
            LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
            LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
            {link_ms1}
            LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
            LEFT JOIN SCORE_IPF ON SCORE_IPF.FEATURE_ID = FEATURE.ID
            INNER JOIN PEPTIDE AS PEPTIDE_IPF ON SCORE_IPF.PEPTIDE_ID = PEPTIDE_IPF.ID
            WHERE SCORE_MS2.QVALUE < {cfg.max_rs_peakgroup_qvalue} AND SCORE_IPF.PEP < {cfg.ipf_max_peptidoform_pep}
            ORDER BY transition_group_id, peak_group_rank;
        """
        return pd.read_sql_query(query, con)

    def _read_augmented_data(self, con, cfg):
        """Read standard data augmented with IPF information."""
        score_ms1_pep, link_ms1 = self._get_ms1_score_info(con)

        query = f"""
            SELECT RUN.ID AS id_run,
                  PEPTIDE.ID AS id_peptide,
                  PRECURSOR.ID AS transition_group_id,
                  PRECURSOR.DECOY AS decoy,
                  RUN.ID AS run_id,
                  RUN.FILENAME AS filename,
                  FEATURE.EXP_RT AS RT,
                  FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
                  FEATURE.DELTA_RT AS delta_rt,
                  FEATURE.NORM_RT AS iRT,
                  PRECURSOR.LIBRARY_RT AS assay_iRT,
                  FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT,
                  FEATURE.ID AS id,
                  PEPTIDE.UNMODIFIED_SEQUENCE AS Sequence,
                  PEPTIDE.MODIFIED_SEQUENCE AS FullPeptideName,
                  PRECURSOR.CHARGE AS Charge,
                  PRECURSOR.PRECURSOR_MZ AS mz,
                  FEATURE_MS2.AREA_INTENSITY AS Intensity,
                  FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
                  FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
                  FEATURE.LEFT_WIDTH AS leftWidth,
                  FEATURE.RIGHT_WIDTH AS rightWidth,
                  SCORE_MS2.RANK AS peak_group_rank,
                  SCORE_MS2.SCORE AS d_score,
                  SCORE_MS2.QVALUE AS m_score,
                  {score_ms1_pep} AS ms1_pep,
                  SCORE_MS2.PEP AS ms2_pep
            FROM PRECURSOR
            INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
            LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
            LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
            {link_ms1}
            LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
            WHERE SCORE_MS2.QVALUE < {cfg.max_rs_peakgroup_qvalue}
            ORDER BY transition_group_id, peak_group_rank;
        """
        data = pd.read_sql_query(query, con)

        # Augment with IPF data
        ipf_data = self._get_ipf_augmentation_data(con, cfg)
        return pd.merge(data, ipf_data, how="left", on="id")

    def _read_standard_data(self, con, cfg):
        """Read standard OpenSWATH data without IPF."""
        query = f"""
            SELECT RUN.ID AS id_run,
                  PEPTIDE.ID AS id_peptide,
                  PRECURSOR.ID AS transition_group_id,
                  PRECURSOR.DECOY AS decoy,
                  RUN.ID AS run_id,
                  RUN.FILENAME AS filename,
                  FEATURE.EXP_RT AS RT,
                  FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
                  FEATURE.DELTA_RT AS delta_rt,
                  FEATURE.NORM_RT AS iRT,
                  PRECURSOR.LIBRARY_RT AS assay_iRT,
                  FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT,
                  FEATURE.ID AS id,
                  PEPTIDE.UNMODIFIED_SEQUENCE AS Sequence,
                  PEPTIDE.MODIFIED_SEQUENCE AS FullPeptideName,
                  PRECURSOR.CHARGE AS Charge,
                  PRECURSOR.PRECURSOR_MZ AS mz,
                  FEATURE_MS2.AREA_INTENSITY AS Intensity,
                  FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
                  FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
                  FEATURE.LEFT_WIDTH AS leftWidth,
                  FEATURE.RIGHT_WIDTH AS rightWidth,
                  SCORE_MS2.RANK AS peak_group_rank,
                  SCORE_MS2.SCORE AS d_score,
                  SCORE_MS2.QVALUE AS m_score
            FROM PRECURSOR
            INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
            LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
            LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
            LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
            WHERE SCORE_MS2.QVALUE < {cfg.max_rs_peakgroup_qvalue}
            ORDER BY transition_group_id, peak_group_rank;
        """
        return pd.read_sql_query(query, con)

    def _augment_data(self, data, con, cfg):
        """Apply common data augmentations to the base dataset."""
        if cfg.transition_quantification:
            logger.info("Adding transition-level quantification data.")
            data = self._add_transition_data(data, con, cfg)

        logger.info("Adding protein information.")
        data = self._add_protein_data(data, con)

        if cfg.peptide:
            logger.info("Adding peptide error rate data.")
            data = self._add_peptide_data(data, con, cfg)

        if cfg.protein:
            logger.info("Adding protein error rate data.")
            data = self._add_protein_error_data(data, con, cfg)

        return data

    def _get_ms1_score_info(self, con):
        """Get MS1 score information if available."""
        if check_sqlite_table(con, "SCORE_MS1"):
            return (
                "SCORE_MS1.PEP",
                "LEFT JOIN SCORE_MS1 ON SCORE_MS1.FEATURE_ID = FEATURE.ID",
            )
        return "NULL", ""

    def _get_ipf_augmentation_data(self, con, cfg):
        """Get IPF data for augmentation."""
        query = f"""
            SELECT FEATURE_ID AS id,
                  MODIFIED_SEQUENCE AS ipf_FullUniModPeptideName,
                  PRECURSOR_PEAKGROUP_PEP AS ipf_precursor_peakgroup_pep,
                  PEP AS ipf_peptidoform_pep,
                  QVALUE AS ipf_peptidoform_m_score
            FROM SCORE_IPF
            INNER JOIN PEPTIDE ON SCORE_IPF.PEPTIDE_ID = PEPTIDE.ID
            WHERE SCORE_IPF.PEP < {cfg.ipf_max_peptidoform_pep};
        """
        data_augmented = pd.read_sql_query(query, con)

        return (
            data_augmented.groupby("id")
            .apply(
                lambda x: pd.Series(
                    {
                        "ipf_FullUniModPeptideName": ";".join(
                            x[
                                x["ipf_peptidoform_pep"]
                                == np.min(x["ipf_peptidoform_pep"])
                            ]["ipf_FullUniModPeptideName"]
                        ),
                        "ipf_precursor_peakgroup_pep": x[
                            x["ipf_peptidoform_pep"] == np.min(x["ipf_peptidoform_pep"])
                        ]["ipf_precursor_peakgroup_pep"].values[0],
                        "ipf_peptidoform_pep": x[
                            x["ipf_peptidoform_pep"] == np.min(x["ipf_peptidoform_pep"])
                        ]["ipf_peptidoform_pep"].values[0],
                        "ipf_peptidoform_m_score": x[
                            x["ipf_peptidoform_pep"] == np.min(x["ipf_peptidoform_pep"])
                        ]["ipf_peptidoform_m_score"].values[0],
                    }
                )
            )
            .reset_index(level="id")
        )

    def _get_base_openswath_data(self, con, cfg):
        """Get base OpenSWATH data without augmentations."""
        query = f"""
            SELECT RUN.ID AS id_run,
                  PEPTIDE.ID AS id_peptide,
                  PRECURSOR.ID AS transition_group_id,
                  PRECURSOR.DECOY AS decoy,
                  RUN.ID AS run_id,
                  RUN.FILENAME AS filename,
                  FEATURE.EXP_RT AS RT,
                  FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
                  FEATURE.DELTA_RT AS delta_rt,
                  FEATURE.NORM_RT AS iRT,
                  PRECURSOR.LIBRARY_RT AS assay_iRT,
                  FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT,
                  FEATURE.ID AS id,
                  PEPTIDE.UNMODIFIED_SEQUENCE AS Sequence,
                  PEPTIDE.MODIFIED_SEQUENCE AS FullPeptideName,
                  PRECURSOR.CHARGE AS Charge,
                  PRECURSOR.PRECURSOR_MZ AS mz,
                  FEATURE_MS2.AREA_INTENSITY AS Intensity,
                  FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
                  FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
                  FEATURE.LEFT_WIDTH AS leftWidth,
                  FEATURE.RIGHT_WIDTH AS rightWidth,
                  SCORE_MS2.RANK AS peak_group_rank,
                  SCORE_MS2.SCORE AS d_score,
                  SCORE_MS2.QVALUE AS m_score
            FROM PRECURSOR
            INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
            LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
            LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
            LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
            WHERE SCORE_MS2.QVALUE < {cfg.max_rs_peakgroup_qvalue}
            ORDER BY transition_group_id, peak_group_rank;
        """
        return pd.read_sql_query(query, con)

    def _add_transition_data(self, data, con, cfg):
        """Add transition-level quantification data."""
        if check_sqlite_table(con, "SCORE_TRANSITION"):
            transition_query = f"""
                SELECT FEATURE_TRANSITION.FEATURE_ID AS id,
                      GROUP_CONCAT(AREA_INTENSITY,';') AS aggr_Peak_Area,
                      GROUP_CONCAT(APEX_INTENSITY,';') AS aggr_Peak_Apex,
                      GROUP_CONCAT(TRANSITION.ID || "_" || TRANSITION.TYPE || TRANSITION.ORDINAL || "_" || TRANSITION.CHARGE,';') AS aggr_Fragment_Annotation
                FROM FEATURE_TRANSITION
                INNER JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                INNER JOIN SCORE_TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = SCORE_TRANSITION.TRANSITION_ID AND FEATURE_TRANSITION.FEATURE_ID = SCORE_TRANSITION.FEATURE_ID
                WHERE TRANSITION.DECOY == 0 AND SCORE_TRANSITION.PEP < {cfg.max_transition_pep}
                GROUP BY FEATURE_TRANSITION.FEATURE_ID
            """
        else:
            transition_query = """
                SELECT FEATURE_ID AS id,
                      GROUP_CONCAT(AREA_INTENSITY,';') AS aggr_Peak_Area,
                      GROUP_CONCAT(APEX_INTENSITY,';') AS aggr_Peak_Apex,
                      GROUP_CONCAT(TRANSITION.ID || "_" || TRANSITION.TYPE || TRANSITION.ORDINAL || "_" || TRANSITION.CHARGE,';') AS aggr_Fragment_Annotation
                FROM FEATURE_TRANSITION
                INNER JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                GROUP BY FEATURE_ID
            """

        data_transition = pd.read_sql_query(transition_query, con)
        return pd.merge(data, data_transition, how="left", on=["id"])

    def _add_protein_data(self, data, con):
        """Add protein identifier data."""
        data_protein = pd.read_sql_query(
            """
            SELECT PEPTIDE_ID AS id_peptide,
                  GROUP_CONCAT(PROTEIN.PROTEIN_ACCESSION,';') AS ProteinName
            FROM PEPTIDE_PROTEIN_MAPPING
            INNER JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
            GROUP BY PEPTIDE_ID;
        """,
            con,
        )
        return pd.merge(data, data_protein, how="inner", on=["id_peptide"])

    def _add_peptide_data(self, data, con, cfg):
        """Add peptide-level error rate data."""
        if not check_sqlite_table(con, "SCORE_PEPTIDE"):
            return data

        # Add run-specific peptide data
        data_peptide_run = pd.read_sql_query(
            """
            SELECT RUN_ID AS id_run,
                  PEPTIDE_ID AS id_peptide,
                  QVALUE AS m_score_peptide_run_specific
            FROM SCORE_PEPTIDE
            WHERE CONTEXT == 'run-specific';
        """,
            con,
        )
        if len(data_peptide_run.index) > 0:
            data = pd.merge(
                data, data_peptide_run, how="inner", on=["id_run", "id_peptide"]
            )

        # Add experiment-wide peptide data
        data_peptide_experiment = pd.read_sql_query(
            """
            SELECT RUN_ID AS id_run,
                  PEPTIDE_ID AS id_peptide,
                  QVALUE AS m_score_peptide_experiment_wide
            FROM SCORE_PEPTIDE
            WHERE CONTEXT == 'experiment-wide';
        """,
            con,
        )
        if len(data_peptide_experiment.index) > 0:
            data = pd.merge(data, data_peptide_experiment, on=["id_run", "id_peptide"])

        # Add global peptide data
        data_peptide_global = pd.read_sql_query(
            """
            SELECT PEPTIDE_ID AS id_peptide,
                  QVALUE AS m_score_peptide_global
            FROM SCORE_PEPTIDE
            WHERE CONTEXT == 'global';
        """,
            con,
        )
        if len(data_peptide_global.index) > 0:
            data = pd.merge(
                data,
                data_peptide_global[
                    data_peptide_global["m_score_peptide_global"]
                    < cfg.max_global_peptide_qvalue
                ],
                on=["id_peptide"],
            )

        return data

    def _add_protein_error_data(self, data, con, cfg):
        """Add protein-level error rate data."""
        if not check_sqlite_table(con, "SCORE_PROTEIN"):
            return data

        # Add run-specific protein data
        data_protein_run = pd.read_sql_query(
            """
            SELECT RUN_ID AS id_run,
                  PEPTIDE_ID AS id_peptide,
                  MIN(QVALUE) AS m_score_protein_run_specific
            FROM PEPTIDE_PROTEIN_MAPPING
            INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID
            WHERE CONTEXT == 'run-specific'
            GROUP BY RUN_ID, PEPTIDE_ID;
        """,
            con,
        )
        if len(data_protein_run.index) > 0:
            data = pd.merge(
                data, data_protein_run, how="inner", on=["id_run", "id_peptide"]
            )

        # Add experiment-wide protein data
        data_protein_experiment = pd.read_sql_query(
            """
            SELECT RUN_ID AS id_run,
                  PEPTIDE_ID AS id_peptide,
                  MIN(QVALUE) AS m_score_protein_experiment_wide
            FROM PEPTIDE_PROTEIN_MAPPING
            INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID
            WHERE CONTEXT == 'experiment-wide'
            GROUP BY RUN_ID, PEPTIDE_ID;
        """,
            con,
        )
        if len(data_protein_experiment.index) > 0:
            data = pd.merge(
                data,
                data_protein_experiment,
                how="inner",
                on=["id_run", "id_peptide"],
            )

        # Add global protein data
        data_protein_global = pd.read_sql_query(
            """
            SELECT PEPTIDE_ID AS id_peptide,
                  MIN(QVALUE) AS m_score_protein_global
            FROM PEPTIDE_PROTEIN_MAPPING
            INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID
            WHERE CONTEXT == 'global'
            GROUP BY PEPTIDE_ID;
        """,
            con,
        )
        if len(data_protein_global.index) > 0:
            data = pd.merge(
                data,
                data_protein_global[
                    data_protein_global["m_score_protein_global"]
                    < cfg.max_global_protein_qvalue
                ],
                how="inner",
                on=["id_peptide"],
            )

        return data


class OSWWriter(BaseOSWWriter):
    """
    Class for writing OpenSWATH results to various formats.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)

    def export(self) -> None:
        """Main entry point for writing data based on configured format"""
        if self.config.export_format in ["parquet", "parquet_split"]:
            self._write_parquet()
        else:
            raise ValueError(
                f"Unsupported OSW export format: {self.config.export_format}. "
                "Supported formats are 'parquet' and 'parquet_split'."
            )

    def _write_parquet(self) -> None:
        """Handle parquet export based on configuration"""
        if self.config.file_type != "osw":
            raise ValueError("Parquet export only supported from OSW files")

        if self.config.export_format == "parquet_split":
            self._convert_to_split_parquet()
        else:
            self._convert_to_single_parquet()

    def _convert_to_split_parquet(self) -> None:
        """Convert OSW to split parquet format"""
        conn = duckdb.connect(":memory:")
        load_sqlite_scanner(conn)

        try:
            # Prepare column information
            column_info = self._prepare_column_info(conn)

            if self.config.split_runs:
                self._export_split_by_run(conn, column_info)
            else:
                self._export_combined(conn, column_info)

        finally:
            conn.close()

    def _convert_to_single_parquet(self) -> None:
        """Convert OSW to single parquet file"""
        conn = duckdb.connect(":memory:")
        load_sqlite_scanner(conn)

        try:
            # Prepare column information
            column_info = self._prepare_column_info(conn)
            self._export_single_file(conn, column_info)
        finally:
            conn.close()

    def _prepare_column_info(self, conn) -> dict:
        """Prepare column information and table checks"""
        with sqlite3.connect(self.config.infile) as sql_conn:
            table_names = set(
                row[0]
                for row in sql_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            )

            column_info = {
                "gene_tables_exist": {"PEPTIDE_GENE_MAPPING", "GENE"}.issubset(
                    table_names
                ),
                "precursor_columns": get_table_columns(self.config.infile, "PRECURSOR"),
                "transition_columns": get_table_columns(
                    self.config.infile, "TRANSITION"
                ),
                "feature_columns": get_table_columns(self.config.infile, "FEATURE"),
                "feature_ms2_alignment_exists": check_sqlite_table(
                    sql_conn, "FEATURE_MS2_ALIGNMENT"
                ),
                "has_library_drift_time": "LIBRARY_DRIFT_TIME"
                in get_table_columns(self.config.infile, "PRECURSOR"),
                "has_annotation": "ANNOTATION"
                in get_table_columns(self.config.infile, "TRANSITION"),
                "has_im": "EXP_IM" in get_table_columns(self.config.infile, "FEATURE"),
                "feature_ms1_cols": [
                    col
                    for col in get_table_columns_with_types(
                        self.config.infile, "FEATURE_MS1"
                    )
                    if col[0] != "FEATURE_ID"
                ],
                "feature_ms2_cols": [
                    col
                    for col in get_table_columns_with_types(
                        self.config.infile, "FEATURE_MS2"
                    )
                    if col[0] != "FEATURE_ID"
                ],
                "feature_transition_cols": [
                    col
                    for col in get_table_columns_with_types(
                        self.config.infile, "FEATURE_TRANSITION"
                    )
                    if col[0] not in ["FEATURE_ID", "TRANSITION_ID"]
                ],
                "score_ms1_exists": {"SCORE_MS1"}.issubset(table_names),
                "score_ms2_exists": {"SCORE_MS2"}.issubset(table_names),
                "score_ipf_exists": {"SCORE_IPF"}.issubset(table_names),
                "score_peptide_exists": {"SCORE_PEPTIDE"}.issubset(table_names),
                "score_protein_exists": {"SCORE_PROTEIN"}.issubset(table_names),
                "score_gene_exists": {"SCORE_GENE"}.issubset(table_names),
                "score_transition_exists": {"SCORE_TRANSITION"}.issubset(table_names),
            }

            if column_info["score_protein_exists"]:
                logger.debug("Checking SCORE_PROTEIN contexts")
                column_info['score_protein_contexts'] = self._check_contexts(sql_conn, "SCORE_PROTEIN")
            if column_info["score_peptide_exists"]:
                logger.debug("Checking SCORE_PEPTIDE contexts")
                column_info['score_peptide_contexts'] = self._check_contexts(sql_conn, "SCORE_PEPTIDE")

        return column_info

    def _export_split_by_run(self, conn, column_info: dict) -> None:
        """Export data split by run into separate directories"""
        os.makedirs(self.config.outfile, exist_ok=True)

        # Get run information
        run_df = conn.execute(
            f"SELECT ID, FILENAME FROM sqlite_scan('{self.config.infile}', 'RUN')"
        ).fetchdf()
        run_df["BASENAME"] = run_df["FILENAME"].apply(
            lambda x: re.sub(r"(\.[^.]*)*$", "", os.path.basename(x))
        )
        logger.info(f"Found {len(run_df)} runs to export.")

        # Export each run
        for _, row in run_df.iterrows():
            run_id = row["ID"]
            run_name = row["BASENAME"]
            run_dir = os.path.join(self.config.outfile, f"{run_name}.oswpq")
            os.makedirs(run_dir, exist_ok=True)

            logger.info(f"Exporting run: {run_name} to {run_dir}")

            # Export precursor data
            precursor_path = os.path.join(run_dir, "precursors_features.parquet")
            precursor_query = (
                self._build_precursor_query(conn, column_info)
                + f"\nWHERE FEATURE.RUN_ID = {run_id}"
            )
            logger.info(f"Exporting precursor data to {precursor_path}")
            self._execute_copy_query(conn, precursor_query, precursor_path)

            # Export transition data
            transition_path = os.path.join(run_dir, "transition_features.parquet")
            transition_query_run = (
                self._build_transition_query(column_info)
                + f"\nWHERE FEATURE.RUN_ID = {run_id}"
            )
            transition_query_null = (
                self._build_transition_query(column_info)
                + "\nWHERE FEATURE.RUN_ID IS NULL"
            )
            combined_transition_query = (
                f"{transition_query_run}\nUNION ALL\n{transition_query_null}"
            )
            logger.info(f"Exporting transition data to {transition_path}")
            self._execute_copy_query(conn, combined_transition_query, transition_path)

        # Export alignment data if exists
        if column_info["feature_ms2_alignment_exists"]:
            logger.info("Exporting alignment data for all runs")
            self._export_alignment_data(conn)

    def _export_combined(self, conn, column_info: dict) -> None:
        """Export combined data (all runs together)"""
        os.makedirs(self.config.outfile, exist_ok=True)

        # Export precursor data
        precursor_path = os.path.join(
            self.config.outfile, "precursors_features.parquet"
        )
        logger.info(f"Exporting precursor data to {precursor_path}")
        precursor_query = self._build_precursor_query(conn, column_info)
        self._execute_copy_query(conn, precursor_query, precursor_path)

        # Export transition data
        transition_path = os.path.join(
            self.config.outfile, "transition_features.parquet"
        )
        logger.info(f"Exporting transition data to {transition_path}")
        transition_query = self._build_transition_query(column_info)
        self._execute_copy_query(conn, transition_query, transition_path)

        # Export alignment data if exists
        if column_info["feature_ms2_alignment_exists"]:
            logger.info("Exporting alignment data")
            self._export_alignment_data(conn)

    def _export_single_file(self, conn, column_info: dict) -> None:
        """Export all data to a single parquet file"""
        # Create temp table with combined schema
        logger.debug("Creating temporary table for combined export")
        self._create_temp_table(conn, column_info)

        # Insert precursor data
        logger.debug("Inserting precursor data into temp table")
        precursor_query = self._build_combined_precursor_query(conn, column_info)
        conn.execute(f"INSERT INTO temp_table {precursor_query}")

        # Insert transition data
        logger.debug("Inserting transition data into temp table")
        transition_query = self._build_combined_transition_query(column_info)
        conn.execute(f"INSERT INTO temp_table {transition_query}")

        # Export to parquet
        logger.info(f"Exporting combined data to {self.config.outfile}")
        self._execute_copy_query(conn, "SELECT * FROM temp_table", self.config.outfile)

        # Export alignment data if exists
        if column_info["feature_ms2_alignment_exists"]:
            alignment_path = (
                os.path.splitext(self.config.outfile)[0] + "_feature_alignment.parquet"
            )
            logger.info(f"Exporting alignment data to {alignment_path}")
            self._export_alignment_data(conn, alignment_path)

    def _build_precursor_query(self, conn, column_info: dict) -> str:
        """Build SQL query for precursor data"""
        feature_ms1_cols_sql = ", ".join(
            f"FEATURE_MS1.{col[0]} AS FEATURE_MS1_{col[0]}"
            for col in column_info["feature_ms1_cols"]
        )

        feature_ms2_cols_sql = ", ".join(
            f"FEATURE_MS2.{col[0]} AS FEATURE_MS2_{col[0]}"
            for col in column_info["feature_ms2_cols"]
        )

        # Check if score tables exist and build score SQLs
        score_cols_selct, score_table_joins, score_column_views = (
            self._build_score_column_selection_and_joins(column_info)
        )

        # First get the peptide table and process it with pyopenms
        logger.info("Generating peptide unimod to codename mapping")
        with sqlite3.connect(self.config.infile) as sql_conn:
            peptide_df = pd.read_sql_query(
                "SELECT ID, MODIFIED_SEQUENCE FROM PEPTIDE", sql_conn
            )
        peptide_df["codename"] = peptide_df["MODIFIED_SEQUENCE"].apply(
            unimod_to_codename
        )

        # Create the merged mapping 
        unimod_mask = peptide_df["MODIFIED_SEQUENCE"].str.contains("UniMod")
        merged_df = pd.merge(
            peptide_df[unimod_mask][["codename", "ID"]],
            peptide_df[~unimod_mask][["codename", "ID"]],
            on="codename",
            suffixes=("_unimod", "_codename"),
            how="outer",
        )

        # Fill NaN values in the 'ID_codename' column with the 'ID_unimod' values
        merged_df["ID_codename"] = merged_df["ID_codename"].fillna(
            merged_df["ID_unimod"]
        )
        # Fill NaN values in the 'ID_unimod' column with the 'ID_codename' values
        merged_df["ID_unimod"] = merged_df["ID_unimod"].fillna(merged_df["ID_codename"])

        merged_df["ID_unimod"] = merged_df["ID_unimod"].astype(int)
        merged_df["ID_codename"] = merged_df["ID_codename"].astype(int)

        # Register peptide_ipf_map
        conn.register(
            "peptide_ipf_map",
            merged_df.rename(
                columns={"ID_unimod": "PEPTIDE_ID", "ID_codename": "IPF_PEPTIDE_ID"}
            ),
        )

        return f"""
            -- Need to map the unimod peptide ids to the ipf codename peptide ids. The section below is commented out, since it's limited to only the 4 common modifications. Have replaced it above with a more general approach that handles all modifications using pyopenms
            --WITH normalized_peptides AS (
            --    SELECT 
            --        ID AS PEPTIDE_ID,
            --        REPLACE(
            --            REPLACE(
            --                REPLACE(
            --                    REPLACE(MODIFIED_SEQUENCE, '(UniMod:1)', '(Acetyl)'),
            --                '(UniMod:35)', '(Oxidation)'),
            --            '(UniMod:21)', '(Phospho)'),
            --        '(UniMod:4)', '(Carbamidomethyl)') AS NORMALIZED_SEQUENCE
            --    FROM sqlite_scan('{self.config.infile}', 'PEPTIDE')
            --),
            --ipf_groups AS (
            --    SELECT 
            --        NORMALIZED_SEQUENCE,
            --        MIN(PEPTIDE_ID) AS IPF_PEPTIDE_ID
            --    FROM normalized_peptides
            --    GROUP BY NORMALIZED_SEQUENCE
            --),
            --peptide_ipf_map AS (
            --    SELECT 
            --        np.PEPTIDE_ID,
            --        g.IPF_PEPTIDE_ID
            --    FROM normalized_peptides np
            --    JOIN ipf_groups g USING (NORMALIZED_SEQUENCE)
            --) 

            {score_column_views}
            SELECT 
                PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID AS PROTEIN_ID,
                PEPTIDE.ID AS PEPTIDE_ID,
                pipf.IPF_PEPTIDE_ID AS IPF_PEPTIDE_ID,
                PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID AS PRECURSOR_ID,
                PROTEIN.PROTEIN_ACCESSION AS PROTEIN_ACCESSION,
                PEPTIDE.UNMODIFIED_SEQUENCE,
                PEPTIDE.MODIFIED_SEQUENCE,
                PRECURSOR.TRAML_ID AS PRECURSOR_TRAML_ID,
                PRECURSOR.GROUP_LABEL AS PRECURSOR_GROUP_LABEL,
                PRECURSOR.PRECURSOR_MZ AS PRECURSOR_MZ,
                PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
                PRECURSOR.LIBRARY_INTENSITY AS PRECURSOR_LIBRARY_INTENSITY,
                PRECURSOR.LIBRARY_RT AS PRECURSOR_LIBRARY_RT,
                {"PRECURSOR.LIBRARY_DRIFT_TIME" if column_info["has_library_drift_time"] else "NULL"} AS PRECURSOR_LIBRARY_DRIFT_TIME,
                {"PEPTIDE_GENE_MAPPING.GENE_ID" if column_info["gene_tables_exist"] else "NULL"} AS GENE_ID,
                {"GENE.GENE_NAME" if column_info["gene_tables_exist"] else "NULL"} AS GENE_NAME,
                {"GENE.DECOY" if column_info["gene_tables_exist"] else "NULL"} AS GENE_DECOY,
                PROTEIN.DECOY AS PROTEIN_DECOY,
                PEPTIDE.DECOY AS PEPTIDE_DECOY,
                PRECURSOR.DECOY AS PRECURSOR_DECOY,
                FEATURE.RUN_ID AS RUN_ID,
                RUN.FILENAME,
                FEATURE.ID AS FEATURE_ID,
                FEATURE.EXP_RT,
                {"FEATURE.EXP_IM" if column_info["has_im"] else "NULL"} AS EXP_IM,
                FEATURE.NORM_RT,
                FEATURE.DELTA_RT,
                FEATURE.LEFT_WIDTH,
                FEATURE.RIGHT_WIDTH,
                {feature_ms1_cols_sql},
                {feature_ms2_cols_sql},
                {score_cols_selct}
            FROM sqlite_scan('{self.config.infile}', 'PRECURSOR') AS PRECURSOR
            INNER JOIN sqlite_scan('{self.config.infile}', 'PRECURSOR_PEPTIDE_MAPPING') AS PRECURSOR_PEPTIDE_MAPPING 
                ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'PEPTIDE') AS PEPTIDE 
                ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            INNER JOIN peptide_ipf_map AS pipf
                ON PEPTIDE.ID = pipf.PEPTIDE_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'PEPTIDE_PROTEIN_MAPPING') AS PEPTIDE_PROTEIN_MAPPING 
                ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'PROTEIN') AS PROTEIN 
                ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
            {self._build_gene_joins(column_info)}
            INNER JOIN sqlite_scan('{self.config.infile}', 'FEATURE') AS FEATURE 
                ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'FEATURE_MS1') AS FEATURE_MS1 
                ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'FEATURE_MS2') AS FEATURE_MS2 
                ON FEATURE.ID = FEATURE_MS2.FEATURE_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'RUN') AS RUN 
                ON FEATURE.RUN_ID = RUN.ID
            {score_table_joins}
            """

    def _build_transition_query(self, column_info: dict) -> str:
        """Build SQL query for transition data"""
        feature_transition_cols_sql = ", ".join(
            f"FEATURE_TRANSITION.{col[0]} AS FEATURE_TRANSITION_{col[0]}"
            for col in column_info["feature_transition_cols"]
        )

        annotation = (
            "TRANSITION.ANNOTATION"
            if column_info["has_annotation"]
            else "TRANSITION.TYPE || CAST(TRANSITION.ORDINAL AS VARCHAR) || '^' || CAST(TRANSITION.CHARGE AS VARCHAR)"
        )

        return f"""
            SELECT 
                FEATURE.RUN_ID AS RUN_ID,
                TRANSITION_PEPTIDE_MAPPING.PEPTIDE_ID AS IPF_PEPTIDE_ID,
                TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID AS PRECURSOR_ID,
                TRANSITION.ID AS TRANSITION_ID,
                TRANSITION.TRAML_ID AS TRANSITION_TRAML_ID,
                TRANSITION.PRODUCT_MZ,
                TRANSITION.CHARGE AS TRANSITION_CHARGE,
                TRANSITION.TYPE AS TRANSITION_TYPE,
                TRANSITION.ORDINAL AS TRANSITION_ORDINAL,
                {annotation} AS ANNOTATION,
                TRANSITION.DETECTING AS TRANSITION_DETECTING,
                TRANSITION.LIBRARY_INTENSITY AS TRANSITION_LIBRARY_INTENSITY,
                TRANSITION.DECOY AS TRANSITION_DECOY,
                FEATURE.ID AS FEATURE_ID,
                {feature_transition_cols_sql}
            FROM sqlite_scan('{self.config.infile}', 'TRANSITION') AS TRANSITION
            FULL JOIN sqlite_scan('{self.config.infile}', 'TRANSITION_PRECURSOR_MAPPING') AS TRANSITION_PRECURSOR_MAPPING 
                ON TRANSITION.ID = TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID
            FULL JOIN sqlite_scan('{self.config.infile}', 'TRANSITION_PEPTIDE_MAPPING') AS TRANSITION_PEPTIDE_MAPPING 
                ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
            FULL JOIN sqlite_scan('{self.config.infile}', 'FEATURE_TRANSITION') AS FEATURE_TRANSITION 
                ON TRANSITION.ID = FEATURE_TRANSITION.TRANSITION_ID
            FULL JOIN (
                SELECT ID, RUN_ID
                FROM sqlite_scan('{self.config.infile}', 'FEATURE')
            ) AS FEATURE
                ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
            """

    def _build_combined_precursor_query(self, conn, column_info: dict) -> str:
        """Build combined precursor query for single file export"""
        feature_ms1_cols_sql = ", ".join(
            f"FEATURE_MS1.{col[0]} AS FEATURE_MS1_{col[0]}"
            for col in column_info["feature_ms1_cols"]
        )

        feature_ms2_cols_sql = ", ".join(
            f"FEATURE_MS2.{col[0]} AS FEATURE_MS2_{col[0]}"
            for col in column_info["feature_ms2_cols"]
        )

        as_null_feature_transition_cols_sql = ", ".join(
            f"NULL AS FEATURE_TRANSITION_{col[0]}"
            for col in column_info["feature_transition_cols"]
        )

        # First get the peptide table and process it with pyopenms
        logger.info("Generating peptide unimod to codename mapping")
        with sqlite3.connect(self.config.infile) as sql_conn:
            peptide_df = pd.read_sql_query(
                "SELECT ID, MODIFIED_SEQUENCE FROM PEPTIDE", sql_conn
            )
        peptide_df["codename"] = peptide_df["MODIFIED_SEQUENCE"].apply(
            unimod_to_codename
        )

        # Create the merged mapping as you did in your example
        unimod_mask = peptide_df["MODIFIED_SEQUENCE"].str.contains("UniMod")
        merged_df = pd.merge(
            peptide_df[unimod_mask][["codename", "ID"]],
            peptide_df[~unimod_mask][["codename", "ID"]],
            on="codename",
            suffixes=("_unimod", "_codename"),
            how="outer",
        )

        # Fill NaN values in the 'ID_codename' column with the 'ID_unimod' values
        merged_df["ID_codename"] = merged_df["ID_codename"].fillna(
            merged_df["ID_unimod"]
        )
        # Fill NaN values in the 'ID_unimod' column with the 'ID_codename' values
        merged_df["ID_unimod"] = merged_df["ID_unimod"].fillna(merged_df["ID_codename"])

        merged_df["ID_unimod"] = merged_df["ID_unimod"].astype(int)
        merged_df["ID_codename"] = merged_df["ID_codename"].astype(int)

        # Register peptide_ipf_map
        conn.register(
            "peptide_ipf_map",
            merged_df.rename(
                columns={"ID_unimod": "PEPTIDE_ID", "ID_codename": "IPF_PEPTIDE_ID"}
            ),
        )

        return f"""
            -- Need to map the unimod peptide ids to the ipf codename peptide ids. The section below is commented out, since it's limited to only the 4 common modifications. Have replaced it above with a more general approach that handles all modifications using pyopenms
            --WITH normalized_peptides AS (
            --    SELECT 
            --        ID AS PEPTIDE_ID,
            --        REPLACE(
            --            REPLACE(
            --                REPLACE(
            --                    REPLACE(MODIFIED_SEQUENCE, '(UniMod:1)', '(Acetyl)'),
            --                '(UniMod:35)', '(Oxidation)'),
            --            '(UniMod:21)', '(Phospho)'),
            --        '(UniMod:4)', '(Carbamidomethyl)') AS NORMALIZED_SEQUENCE
            --    FROM sqlite_scan('{self.config.infile}', 'PEPTIDE')
            --),
            --ipf_groups AS (
            --    SELECT 
            --        NORMALIZED_SEQUENCE,
            --        MIN(PEPTIDE_ID) AS IPF_PEPTIDE_ID
            --    FROM normalized_peptides
            --    GROUP BY NORMALIZED_SEQUENCE
            --),
            --peptide_ipf_map AS (
            --    SELECT 
            --        np.PEPTIDE_ID,
            --        g.IPF_PEPTIDE_ID
            --    FROM normalized_peptides np
            --    JOIN ipf_groups g USING (NORMALIZED_SEQUENCE)
            --) 
            
            SELECT 
                PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID AS PROTEIN_ID,
                PEPTIDE.ID AS PEPTIDE_ID,
                pipf.IPF_PEPTIDE_ID AS IPF_PEPTIDE_ID,
                PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID AS PRECURSOR_ID,
                PROTEIN.PROTEIN_ACCESSION AS PROTEIN_ACCESSION,
                PEPTIDE.UNMODIFIED_SEQUENCE,
                PEPTIDE.MODIFIED_SEQUENCE,
                PRECURSOR.TRAML_ID AS PRECURSOR_TRAML_ID,
                PRECURSOR.GROUP_LABEL AS PRECURSOR_GROUP_LABEL,
                PRECURSOR.PRECURSOR_MZ AS PRECURSOR_MZ,
                PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
                PRECURSOR.LIBRARY_INTENSITY AS PRECURSOR_LIBRARY_INTENSITY,
                PRECURSOR.LIBRARY_RT AS PRECURSOR_LIBRARY_RT,
                {"PRECURSOR.LIBRARY_DRIFT_TIME" if column_info["has_library_drift_time"] else "NULL"} AS PRECURSOR_LIBRARY_DRIFT_TIME,
                {"PEPTIDE_GENE_MAPPING.GENE_ID" if column_info["gene_tables_exist"] else "NULL"} AS GENE_ID,
                {"GENE.GENE_NAME" if column_info["gene_tables_exist"] else "NULL"} AS GENE_NAME,
                {"GENE.DECOY" if column_info["gene_tables_exist"] else "NULL"} AS GENE_DECOY,
                PROTEIN.DECOY AS PROTEIN_DECOY,
                PEPTIDE.DECOY AS PEPTIDE_DECOY,
                PRECURSOR.DECOY AS PRECURSOR_DECOY,
                FEATURE.RUN_ID AS RUN_ID,
                RUN.FILENAME,
                FEATURE.ID AS FEATURE_ID,
                FEATURE.EXP_RT,
                {"FEATURE.EXP_IM" if column_info["has_im"] else "NULL"} AS EXP_IM,
                FEATURE.NORM_RT,
                FEATURE.DELTA_RT,
                FEATURE.LEFT_WIDTH,
                FEATURE.RIGHT_WIDTH,
                {feature_ms1_cols_sql},
                {feature_ms2_cols_sql},
                NULL AS TRANSITION_ID,
                NULL AS TRANSITION_TRAML_ID,
                NULL AS PRODUCT_MZ,
                NULL AS TRANSITION_CHARGE,
                NULL AS TRANSITION_TYPE,
                NULL AS TRANSITION_ORDINAL,
                NULL AS ANNOTATION,
                NULL AS TRANSITION_DETECTING,
                NULL AS TRANSITION_LIBRARY_INTENSITY,
                NULL AS TRANSITION_DECOY,
                {as_null_feature_transition_cols_sql}
            FROM sqlite_scan('{self.config.infile}', 'PRECURSOR') AS PRECURSOR
            INNER JOIN sqlite_scan('{self.config.infile}', 'PRECURSOR_PEPTIDE_MAPPING') AS PRECURSOR_PEPTIDE_MAPPING 
                ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'PEPTIDE') AS PEPTIDE 
                ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            INNER JOIN peptide_ipf_map AS pipf
                ON PEPTIDE.ID = pipf.PEPTIDE_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'PEPTIDE_PROTEIN_MAPPING') AS PEPTIDE_PROTEIN_MAPPING 
                ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'PROTEIN') AS PROTEIN 
                ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
            {self._build_gene_joins(column_info)}
            INNER JOIN sqlite_scan('{self.config.infile}', 'FEATURE') AS FEATURE 
                ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'FEATURE_MS1') AS FEATURE_MS1 
                ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'FEATURE_MS2') AS FEATURE_MS2 
                ON FEATURE.ID = FEATURE_MS2.FEATURE_ID
            INNER JOIN sqlite_scan('{self.config.infile}', 'RUN') AS RUN 
                ON FEATURE.RUN_ID = RUN.ID
            """

    def _build_combined_transition_query(self, column_info: dict) -> str:
        """Build combined transition query for single file export"""
        as_null_feature_ms1_cols_sql = ", ".join(
            f"NULL AS FEATURE_MS1_{col[0]}" for col in column_info["feature_ms1_cols"]
        )

        as_null_feature_ms2_cols_sql = ", ".join(
            f"NULL AS FEATURE_MS2_{col[0]}" for col in column_info["feature_ms2_cols"]
        )

        feature_transition_cols_sql = ", ".join(
            f"FEATURE_TRANSITION.{col[0]} AS FEATURE_TRANSITION_{col[0]}"
            for col in column_info["feature_transition_cols"]
        )

        annotation = (
            "TRANSITION.ANNOTATION"
            if column_info["has_annotation"]
            else "TRANSITION.TYPE || CAST(TRANSITION.ORDINAL AS VARCHAR) || '^' || CAST(TRANSITION.CHARGE AS VARCHAR)"
        )

        return f"""
            SELECT
                NULL AS PROTEIN_ID,
                NULL AS PEPTIDE_ID,
                TRANSITION_PEPTIDE_MAPPING.PEPTIDE_ID AS IPF_PEPTIDE_ID,
                TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID AS PRECURSOR_ID,
                NULL AS PROTEIN_ACCESSION,
                NULL AS UNMODIFIED_SEQUENCE,
                NULL AS MODIFIED_SEQUENCE,
                NULL AS PRECURSOR_TRAML_ID,
                NULL AS PRECURSOR_GROUP_LABEL,
                NULL AS PRECURSOR_MZ,
                NULL AS PRECURSOR_CHARGE,
                NULL AS PRECURSOR_LIBRARY_INTENSITY,
                NULL AS PRECURSOR_LIBRARY_RT,
                NULL AS PRECURSOR_LIBRARY_DRIFT_TIME,
                NULL AS GENE_ID,
                NULL AS GENE_NAME,
                NULL AS GENE_DECOY,
                NULL AS PROTEIN_DECOY,
                NULL AS PEPTIDE_DECOY,
                NULL AS PRECURSOR_DECOY,
                NULL AS RUN_ID,
                NULL AS FILENAME,
                FEATURE_TRANSITION.FEATURE_ID AS FEATURE_ID,
                NULL AS EXP_RT,
                NULL AS EXP_IM,
                NULL as NORM_RT,
                NULL AS DELTA_RT,
                NULL AS LEFT_WIDTH,
                NULL AS RIGHT_WIDTH,
                {as_null_feature_ms1_cols_sql},
                {as_null_feature_ms2_cols_sql},
                TRANSITION.ID AS TRANSITION_ID,
                TRANSITION.TRAML_ID AS TRANSITION_TRAML_ID,
                TRANSITION.PRODUCT_MZ,
                TRANSITION.CHARGE AS TRANSITION_CHARGE,
                TRANSITION.TYPE AS TRANSITION_TYPE,
                TRANSITION.ORDINAL AS TRANSITION_ORDINAL,
                {annotation} AS ANNOTATION,
                TRANSITION.DETECTING AS TRANSITION_DETECTING,
                TRANSITION.LIBRARY_INTENSITY AS TRANSITION_LIBRARY_INTENSITY,
                TRANSITION.DECOY AS TRANSITION_DECOY,
                {feature_transition_cols_sql}
            FROM sqlite_scan('{self.config.infile}', 'TRANSITION_PRECURSOR_MAPPING') AS TRANSITION_PRECURSOR_MAPPING 
            INNER JOIN sqlite_scan('{self.config.infile}', 'TRANSITION') AS TRANSITION
                ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
            FULL JOIN sqlite_scan('{self.config.infile}', 'TRANSITION_PEPTIDE_MAPPING') AS TRANSITION_PEPTIDE_MAPPING 
                ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
            FULL JOIN sqlite_scan('{self.config.infile}', 'FEATURE_TRANSITION') AS FEATURE_TRANSITION 
                ON TRANSITION.ID = FEATURE_TRANSITION.TRANSITION_ID
            """

    def _create_temp_table(self, conn, column_info: dict) -> None:
        """Create temporary table with combined schema"""
        feature_ms1_cols_types = ", ".join(
            f"FEATURE_MS1_{col[0]} {col[1]}" for col in column_info["feature_ms1_cols"]
        )

        feature_ms2_cols_types = ", ".join(
            f"FEATURE_MS2_{col[0]} {col[1]}" for col in column_info["feature_ms2_cols"]
        )

        feature_transition_cols_types = ", ".join(
            f"FEATURE_TRANSITION_{col[0]} {col[1]}"
            for col in column_info["feature_transition_cols"]
        )

        create_temp_table_query = f"""
        CREATE TABLE temp_table (
            PROTEIN_ID BIGINT,
            PEPTIDE_ID BIGINT,
            IPF_PEPTIDE_ID BIGINT,
            PRECURSOR_ID BIGINT,
            PROTEIN_ACCESSION TEXT,
            UNMODIFIED_SEQUENCE TEXT,
            MODIFIED_SEQUENCE TEXT,
            PRECURSOR_TRAML_ID TEXT,
            PRECURSOR_GROUP_LABEL TEXT,
            PRECURSOR_MZ DOUBLE,
            PRECURSOR_CHARGE INTEGER,
            PRECURSOR_LIBRARY_INTENSITY DOUBLE,
            PRECURSOR_LIBRARY_RT DOUBLE,
            PRECURSOR_LIBRARY_DRIFT_TIME DOUBLE,
            GENE_ID BIGINT,
            GENE_NAME TEXT,
            GENE_DECOY BOOLEAN,
            PROTEIN_DECOY BOOLEAN,
            PEPTIDE_DECOY BOOLEAN,
            PRECURSOR_DECOY BOOLEAN,
            RUN_ID BIGINT,
            FILENAME TEXT,
            FEATURE_ID BIGINT,
            EXP_RT DOUBLE,
            EXP_IM DOUBLE,
            NORM_RT DOUBLE,
            DELTA_RT DOUBLE,
            LEFT_WIDTH DOUBLE,
            RIGHT_WIDTH DOUBLE,
            {feature_ms1_cols_types},
            {feature_ms2_cols_types},
            TRANSITION_ID BIGINT,
            TRANSITION_TRAML_ID TEXT,
            PRODUCT_MZ DOUBLE,
            TRANSITION_CHARGE INTEGER,
            TRANSITION_TYPE TEXT,
            TRANSITION_ORDINAL INTEGER,
            ANNOTATION TEXT,
            TRANSITION_DETECTING BOOLEAN,
            TRANSITION_LIBRARY_INTENSITY DOUBLE,
            TRANSITION_DECOY BOOLEAN,
            {feature_transition_cols_types}
        );
        """

        conn.execute(create_temp_table_query)

    def _export_alignment_data(self, conn, path: str = None) -> None:
        """Export feature alignment data"""
        if path is None:
            path = os.path.join(self.config.outfile, "feature_alignment.parquet")

        query = f"""
        SELECT
            ALIGNMENT_ID,
            RUN_ID,
            PRECURSOR_ID,
            ALIGNED_FEATURE_ID AS FEATURE_ID,
            REFERENCE_FEATURE_ID,
            ALIGNED_RT,
            REFERENCE_RT,
            XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
            XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE_TO_REFERENCE, 
            MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE, 
            XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,  
            XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE, 
            MI_TO_ALL AS VAR_MI_TO_ALL, 
            RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION, 
            PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO,
            LABEL AS DECOY
        FROM sqlite_scan('{self.config.infile}', 'FEATURE_MS2_ALIGNMENT')
        """

        self._execute_copy_query(conn, query, path)

    def _build_gene_joins(self, column_info: dict) -> str:
        """Build gene join clauses if gene tables exist"""
        if column_info["gene_tables_exist"]:
            return f"""
                LEFT JOIN sqlite_scan('{self.config.infile}', 'PEPTIDE_GENE_MAPPING') AS PEPTIDE_GENE_MAPPING 
                    ON PEPTIDE.ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID 
                LEFT JOIN sqlite_scan('{self.config.infile}', 'GENE') AS GENE 
                    ON PEPTIDE_GENE_MAPPING.GENE_ID = GENE.ID
            """
        return ""

    def _check_contexts(self, conn, score_table) -> list:
        """Get list of contexts available in score table"""
        contexts_query = f""" SELECT DISTINCT context FROM {score_table} """
        result = conn.execute(contexts_query).fetchall()
        logger.debug("result of contexts query: ", result)
        return [row[0] for row in result]

    def _get_peptide_protein_score_table(self, level, contexts: list) -> str:
        """Create a DuckDB view for peptide/protein score data and return the view name"""
        if level == "peptide":
            id_col = "PEPTIDE_ID"
            score_table = "SCORE_PEPTIDE"
        else:  # level == 'protein'
            id_col = "PROTEIN_ID"
            score_table = "SCORE_PROTEIN"

        view_name = f"{score_table.lower()}_view"

        # Build pivot columns for available contexts
        pivot_cols = []
        for context in contexts:
            # Skip 'global' context as it will be handled separately
            if context != 'global':
                # Convert context to column-safe format: 'run-specific' -> 'RUN_SPECIFIC'
                safe_context = context.upper().replace("-", "_")
                pivot_cols.extend([
                    f"ANY_VALUE(CASE WHEN context = '{context}' THEN SCORE END) as {score_table}_{safe_context}_SCORE",
                    f"ANY_VALUE(CASE WHEN context = '{context}' THEN PVALUE END) as {score_table}_{safe_context}_PVALUE",
                    f"ANY_VALUE(CASE WHEN context = '{context}' THEN QVALUE END) as {score_table}_{safe_context}_QVALUE",
                    f"ANY_VALUE(CASE WHEN context = '{context}' THEN PEP END) as {score_table}_{safe_context}_PEP"
                ])
        
        pivot_cols_str = ", ".join(pivot_cols)
        
        # Non-global contexts pivot query
        if pivot_cols_str:  # Only if there are non-global contexts
            nonGlobal_query = f"""
            SELECT {id_col}, RUN_ID, {pivot_cols_str}
            FROM sqlite_scan('{self.config.infile}', '{score_table}')
            WHERE context != 'global'
            GROUP BY {id_col}, RUN_ID
            """
        else:
            # If no non-global contexts, create empty result with same structure
            nonGlobal_query = f"""
            SELECT {id_col}, RUN_ID
            FROM sqlite_scan('{self.config.infile}', '{score_table}')
            WHERE 1=0
            """

        global_exists = 'global' in contexts
        # Global data query (only if global context exists)
        if global_exists:
            glob_query = f"""
            SELECT {id_col}, 
                SCORE as {score_table}_GLOBAL_SCORE, 
                PVALUE as {score_table}_GLOBAL_PVALUE, 
                QVALUE as {score_table}_GLOBAL_QVALUE, 
                PEP as {score_table}_GLOBAL_PEP
            FROM sqlite_scan('{self.config.infile}', '{score_table}') 
            WHERE context = 'global'
            """
        
        # Build final merged query based on what exists
        if pivot_cols_str and global_exists:
            # Both non-global and global contexts exist
            merged_query = f"""
            SELECT ng.*, 
                g.{score_table}_GLOBAL_SCORE, 
                g.{score_table}_GLOBAL_PVALUE, 
                g.{score_table}_GLOBAL_QVALUE, 
                g.{score_table}_GLOBAL_PEP
            FROM ({nonGlobal_query}) ng
            LEFT JOIN ({glob_query}) g ON ng.{id_col} = g.{id_col}
            """
        elif pivot_cols_str and not global_exists:
            # Only non-global contexts exist
            merged_query = nonGlobal_query
        elif not pivot_cols_str and global_exists:
            # Only global context exists
            merged_query = glob_query
        else:
            # No contexts exist - return empty result with ID columns
            merged_query = f"""
            SELECT {id_col}, RUN_ID
            FROM sqlite_scan('{self.config.infile}', '{score_table}')
            WHERE 1=0
            """

        # Create the view in DuckDB
        return f"{view_name} AS ({merged_query})"
        

    def _build_score_column_selection_and_joins(
        self, column_info: dict
    ) -> Tuple[str, str, str]:
        """Build score column selection and joins based on available score tables"""
        score_columns_to_select = []
        score_tables_to_join = []
        score_views = []
        
        if column_info["score_ms1_exists"]:
            logger.debug("SCORE_MS1 table exists, adding score columns to selection")
            score_columns_to_select.append(
                "SCORE_MS1.SCORE AS SCORE_MS1_SCORE, SCORE_MS1.RANK AS SCORE_MS1_RANK, SCORE_MS1.PVALUE AS SCORE_MS1_P_VALUE, SCORE_MS1.QVALUE AS SCORE_MS1_Q_VALUE, SCORE_MS1.PEP AS SCORE_MS1_PEP"
            )
            score_tables_to_join.append(
                f"INNER JOIN sqlite_scan('{self.config.infile}', 'SCORE_MS1') AS SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID"
            )

        if column_info["score_ms2_exists"]:
            logger.debug("SCORE_MS2 table exists, adding score columns to selection")
            score_columns_to_select.append(
                "SCORE_MS2.SCORE AS SCORE_MS2_SCORE, SCORE_MS2.RANK AS SCORE_MS2_PEAK_GROUP_RANK, SCORE_MS2.PVALUE AS SCORE_MS2_P_VALUE, SCORE_MS2.QVALUE AS SCORE_MS2_Q_VALUE, SCORE_MS2.PEP AS SCORE_MS2_PEP"
            )
            score_tables_to_join.append(
                f"INNER JOIN sqlite_scan('{self.config.infile}', 'SCORE_MS2') AS SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID"
            )

        # Create views for peptide and protein score tables if they exist
        if column_info["score_peptide_exists"]:
            logger.debug("SCORE_PEPTIDE table exists, adding score table view to query")
            score_views.append(self._get_peptide_protein_score_table("peptide", column_info["score_peptide_contexts"]))
            # Add JOIN for peptide score view
            score_tables_to_join.append("INNER JOIN score_peptide_view ON PEPTIDE.ID = score_peptide_view.PEPTIDE_ID AND FEATURE.RUN_ID = score_peptide_view.RUN_ID")
        if column_info["score_protein_exists"]:
            logger.debug("SCORE_PROTEIN table exists, adding score table view to query")
            score_views.append(self._get_peptide_protein_score_table("protein", column_info["score_protein_contexts"]))
            # Add JOIN for protein score view
            score_tables_to_join.append("INNER JOIN score_protein_view ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = score_protein_view.PROTEIN_ID AND FEATURE.RUN_ID = score_protein_view.RUN_ID")

        # Add score columns for peptide and protein contexts
        for table in ["peptide", "protein"]:
            if column_info[f"score_{table}_exists"]:
                logger.debug(f"SCORE_{table.upper()} table exists, adding score columns to selection")
                for context in column_info[f"score_{table}_contexts"]:
                    safe_context = context.upper().replace("-", "_")
                    score_columns_to_select.append(
                        f"score_{table.lower()}_view.SCORE_{table.upper()}_{safe_context}_SCORE AS SCORE_{table.upper()}_{safe_context}_SCORE, "
                        f"score_{table.lower()}_view.SCORE_{table.upper()}_{safe_context}_PVALUE AS SCORE_{table.upper()}_{safe_context}_P_VALUE, "
                        f"score_{table.lower()}_view.SCORE_{table.upper()}_{safe_context}_QVALUE AS SCORE_{table.upper()}_{safe_context}_Q_VALUE, "
                        f"score_{table.lower()}_view.SCORE_{table.upper()}_{safe_context}_PEP AS SCORE_{table.upper()}_{safe_context}_PEP"
                    )
                
        # Properly format the WITH clause
        with_clause = ""
        if score_views:
            with_clause = "WITH " + ", ".join(score_views)
            
        return (
            ", ".join(score_columns_to_select),
            " ".join(score_tables_to_join),
            with_clause
        )

    def _execute_copy_query(self, conn, query: str, path: str) -> None:
        """Execute COPY query with configured compression settings"""
        conn.execute(
            f"COPY ({query}) TO '{path}' "
            f"(FORMAT 'parquet', COMPRESSION '{self.config.compression_method}', "
            f"COMPRESSION_LEVEL {self.config.compression_level})"
        )
