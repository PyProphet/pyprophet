import os
import pickle
from shutil import copyfile
import sqlite3
from typing import Literal
import duckdb
import pandas as pd
import numpy as np
import click
from loguru import logger
from ..util import check_sqlite_table, check_duckdb_table, write_scores_sql_command
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
        if False:
            con = duckdb.connect(self.infile)
            return self._read_duckdb(con)
        else:
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
            return self._read_unscored_data(con)

        ipf_present = self._check_ipf_presence(con, cfg)

        if ipf_present and cfg.ipf == "peptidoform":
            data = self._read_peptidoform_data(con, cfg)
        elif ipf_present and cfg.ipf == "augmented":
            data = self._read_augmented_data(con, cfg)
        else:
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
            data = self._add_transition_data(data, con, cfg)

        data = self._add_protein_data(data, con)

        if cfg.peptide:
            data = self._add_peptide_data(data, con, cfg)

        if cfg.protein:
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
