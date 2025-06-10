import duckdb
import pandas as pd
from loguru import logger

from ..._config import ExportIOConfig
from .._base import BaseParquetReader, BaseParquetWriter
from ..util import get_parquet_column_names


class ParquetReader(BaseParquetReader):
    """
    Class for reading and processing data from an OpenSWATH workflow parquet based file.
    Extended to support exporting functionality.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)

        self._columns = get_parquet_column_names(self.infile)
        self._has_ipf = any(col.startswith("SCORE_IPF_") for col in self._columns)
        self._has_ms1_scores = any(
            col.startswith("SCORE_MS1_") for col in self._columns
        )
        self._has_ms2_scores = any(
            col.startswith("SCORE_MS2_") for col in self._columns
        )
        self._has_transition_scores = any(
            col.startswith("SCORE_TRANSITION_") for col in self._columns
        )

    def read(self) -> pd.DataFrame:
        """
        Main entry point for reading Parquet data.
        """
        con = duckdb.connect()
        try:
            self._init_duckdb_views(con)

            if self._is_unscored_file():
                logger.info("Reading unscored data from Parquet file.")
                return self._read_unscored_data(con)

            if self._has_ipf and self.config.ipf == "peptidoform":
                logger.info("Reading peptidoform IPF data from Parquet file.")
                data = self._read_peptidoform_data(con)
            elif self._has_ipf and self.config.ipf == "augmented":
                logger.info("Reading augmented data with IPF from Parquet file.")
                data = self._read_augmented_data(con)
            else:
                logger.info("Reading standard OpenSWATH data from Parquet file.")
                data = self._read_standard_data(con)

            return self._augment_data(data, con)
        finally:
            con.close()

    def _is_unscored_file(self) -> bool:
        """
        Check if the file is unscored by verifying the presence of the 'SCORE_' columns.
        """
        all_cols = get_parquet_column_names(self.infile)
        return all(not col.startswith("SCORE_") for col in all_cols)

    def _read_unscored_data(self, con) -> pd.DataFrame:
        """
        Read unscored data from Parquet files.
        """
        feature_vars_sql = self._build_feature_vars_sql()

        query = f"""
            SELECT
                RUN_ID AS id_run,
                PEPTIDE_ID AS id_peptide,
                PRECURSOR_ID AS transition_group_id,
                PRECURSOR_DECOY AS decoy,
                RUN_ID AS run_id,
                FILENAME AS filename,
                EXP_RT AS RT,
                EXP_RT - DELTA_RT AS assay_rt,
                DELTA_RT AS delta_rt,
                PRECURSOR_LIBRARY_RT AS assay_RT,
                NORM_RT - PRECURSOR_LIBRARY_RT AS delta_RT,
                FEATURE_ID AS id,
                PRECURSOR_CHARGE AS Charge,
                PRECURSOR_MZ AS mz,
                FEATURE_MS2_AREA_INTENSITY AS Intensity,
                FEATURE_MS1_AREA_INTENSITY AS aggr_prec_Peak_Area,
                FEATURE_MS1_APEX_INTENSITY AS aggr_prec_Peak_Apex,
                LEFT_WIDTH AS leftWidth,
                RIGHT_WIDTH AS rightWidth
                {feature_vars_sql}
            FROM data
            WHERE PROTEIN_ID IS NOT NULL  -- Filter to precursor rows
            ORDER BY transition_group_id
        """
        return con.execute(query).fetchdf()

    def _read_peptidoform_data(self, con) -> pd.DataFrame:
        """
        Read data with peptidoform IPF information.
        """
        score_ms1_pep, _link_ms1 = self._get_ms1_score_info()

        query = f"""
            SELECT
                RUN_ID AS id_run,
                PEPTIDE_ID AS id_peptide,
                (MODIFIED_SEQUENCE || '_' || PRECURSOR_ID) AS transition_group_id,
                PRECURSOR_DECOY AS decoy,
                RUN_ID AS run_id,
                FILENAME AS filename,
                EXP_RT AS RT,
                EXP_RT - DELTA_RT AS assay_rt,
                DELTA_RT AS delta_rt,
                NORM_RT AS iRT,
                PRECURSOR_LIBRARY_RT AS assay_iRT,
                NORM_RT - PRECURSOR_LIBRARY_RT AS delta_iRT,
                FEATURE_ID AS id,
                UNMODIFIED_SEQUENCE AS Sequence,
                MODIFIED_SEQUENCE AS FullPeptideName,
                PRECURSOR_CHARGE AS Charge,
                PRECURSOR_MZ AS mz,
                FEATURE_MS2_AREA_INTENSITY AS Intensity,
                FEATURE_MS1_AREA_INTENSITY AS aggr_prec_Peak_Area,
                FEATURE_MS1_APEX_INTENSITY AS aggr_prec_Peak_Apex,
                LEFT_WIDTH AS leftWidth,
                RIGHT_WIDTH AS rightWidth,
                {score_ms1_pep} AS ms1_pep,
                SCORE_MS2_PEP AS ms2_pep,
                SCORE_IPF_PRECURSOR_PEAKGROUP_PEP AS precursor_pep,
                SCORE_IPF_PEP AS ipf_pep,
                SCORE_MS2_PEAK_GROUP_RANK AS peak_group_rank,
                SCORE_MS2_SCORE AS d_score,
                SCORE_MS2_Q_VALUE AS ms2_m_score,
                SCORE_IPF_QVALUE AS m_score
            FROM data
            WHERE PROTEIN_ID IS NOT NULL
            AND SCORE_MS2_Q_VALUE < {self.config.max_rs_peakgroup_qvalue} 
            AND SCORE_IPF_PEP < {self.config.ipf_max_peptidoform_pep}
            ORDER BY transition_group_id, peak_group_rank
        """
        return con.execute(query).fetchdf()

    def _read_augmented_data(self, con) -> pd.DataFrame:
        """
        Read standard data augmented with IPF information.
        """
        score_ms1_pep, _link_ms1 = self._get_ms1_score_info()

        # First get main data
        query = f"""
            SELECT
                RUN_ID AS id_run,
                PEPTIDE_ID AS id_peptide,
                PRECURSOR_ID AS transition_group_id,
                PRECURSOR_DECOY AS decoy,
                RUN_ID AS run_id,
                FILENAME AS filename,
                EXP_RT AS RT,
                EXP_RT - DELTA_RT AS assay_rt,
                DELTA_RT AS delta_rt,
                NORM_RT AS iRT,
                PRECURSOR_LIBRARY_RT AS assay_iRT,
                NORM_RT - PRECURSOR_LIBRARY_RT AS delta_iRT,
                FEATURE_ID AS id,
                UNMODIFIED_SEQUENCE AS Sequence,
                MODIFIED_SEQUENCE AS FullPeptideName,
                PRECURSOR_CHARGE AS Charge,
                PRECURSOR_MZ AS mz,
                FEATURE_MS2_AREA_INTENSITY AS Intensity,
                FEATURE_MS1_AREA_INTENSITY AS aggr_prec_Peak_Area,
                FEATURE_MS1_APEX_INTENSITY AS aggr_prec_Peak_Apex,
                LEFT_WIDTH AS leftWidth,
                RIGHT_WIDTH AS rightWidth,
                SCORE_MS2_PEAK_GROUP_RANK AS peak_group_rank,
                SCORE_MS2_SCORE AS d_score,
                SCORE_MS2_Q_VALUE AS m_score,
                {score_ms1_pep} AS ms1_pep,
                SCORE_MS2_PEP AS ms2_pep
            FROM data
            WHERE PROTEIN_ID IS NOT NULL
            AND SCORE_MS2_Q_VALUE < {self.config.max_rs_peakgroup_qvalue}
            ORDER BY transition_group_id, peak_group_rank
        """
        data = con.execute(query).fetchdf()

        # Augment with IPF data
        ipf_query = f"""
            SELECT 
                FEATURE_ID AS id,
                MODIFIED_SEQUENCE AS ipf_FullUniModPeptideName,
                SCORE_IPF_PRECURSOR_PEAKGROUP_PEP AS ipf_precursor_peakgroup_pep,
                SCORE_IPF_PEP AS ipf_peptidoform_pep,
                SCORE_IPF_QVALUE AS ipf_peptidoform_m_score
            FROM data
            WHERE SCORE_IPF_PEP < {self.config.ipf_max_peptidoform_pep}
            AND PROTEIN_ID IS NOT NULL
        """
        ipf_data = con.execute(ipf_query).fetchdf()

        # Process IPF data to get best peptidoform per feature
        ipf_data = (
            ipf_data.groupby("id")
            .apply(
                lambda x: pd.Series(
                    {
                        "ipf_FullUniModPeptideName": ";".join(
                            x[
                                x["ipf_peptidoform_pep"]
                                == x["ipf_peptidoform_pep"].min()
                            ]["ipf_FullUniModPeptideName"]
                        ),
                        "ipf_precursor_peakgroup_pep": x.loc[
                            x["ipf_peptidoform_pep"].idxmin(),
                            "ipf_precursor_peakgroup_pep",
                        ],
                        "ipf_peptidoform_pep": x["ipf_peptidoform_pep"].min(),
                        "ipf_peptidoform_m_score": x.loc[
                            x["ipf_peptidoform_pep"].idxmin(), "ipf_peptidoform_m_score"
                        ],
                    }
                )
            )
            .reset_index()
        )

        return pd.merge(data, ipf_data, on="id", how="left")

    def _read_standard_data(self, con) -> pd.DataFrame:
        """
        Read standard OpenSWATH data without IPF.
        """
        query = f"""
            SELECT
                RUN_ID AS id_run,
                PEPTIDE_ID AS id_peptide,
                PRECURSOR_ID AS transition_group_id,
                PRECURSOR_DECOY AS decoy,
                RUN_ID AS run_id,
                FILENAME AS filename,
                EXP_RT AS RT,
                EXP_RT - DELTA_RT AS assay_rt,
                DELTA_RT AS delta_rt,
                NORM_RT AS iRT,
                PRECURSOR_LIBRARY_RT AS assay_iRT,
                NORM_RT - PRECURSOR_LIBRARY_RT AS delta_iRT,
                FEATURE_ID AS id,
                UNMODIFIED_SEQUENCE AS Sequence,
                MODIFIED_SEQUENCE AS FullPeptideName,
                PRECURSOR_CHARGE AS Charge,
                PRECURSOR_MZ AS mz,
                FEATURE_MS2_AREA_INTENSITY AS Intensity,
                FEATURE_MS1_AREA_INTENSITY AS aggr_prec_Peak_Area,
                FEATURE_MS1_APEX_INTENSITY AS aggr_prec_Peak_Apex,
                LEFT_WIDTH AS leftWidth,
                RIGHT_WIDTH AS rightWidth,
                SCORE_MS2_PEAK_GROUP_RANK AS peak_group_rank,
                SCORE_MS2_SCORE AS d_score,
                SCORE_MS2_Q_VALUE AS m_score
            FROM data
            WHERE PROTEIN_ID IS NOT NULL
            AND SCORE_MS2_Q_VALUE < {self.config.max_rs_peakgroup_qvalue}
            ORDER BY transition_group_id, peak_group_rank
        """
        return con.execute(query).fetchdf()

    def _augment_data(self, data, con) -> pd.DataFrame:
        """
        Apply common data augmentations to the base dataset.
        """
        if self.config.transition_quantification:
            logger.info("Adding transition-level quantification data.")
            data = self._add_transition_data(data, con)

        logger.info("Adding protein information.")
        data = self._add_protein_data(data, con)

        if self.config.peptide:
            logger.info("Adding peptide error rate data.")
            data = self._add_peptide_data(data, con)

        if self.config.protein:
            logger.info("Adding protein error rate data.")
            data = self._add_protein_error_data(data, con)

        return data

    def _add_transition_data(self, data, con) -> pd.DataFrame:
        """
        Add transition-level quantification data.
        """
        if self._has_transition_scores:
            query = f"""
                SELECT 
                    FEATURE_ID AS id,
                    STRING_AGG(CAST(FEATURE_TRANSITION_AREA_INTENSITY AS VARCHAR), ';') AS aggr_Peak_Area,
                    STRING_AGG(CAST(FEATURE_TRANSITION_APEX_INTENSITY AS VARCHAR), ';') AS aggr_Peak_Apex,
                    STRING_AGG(TRANSITION_ID || '_' || TRANSITION_TYPE || TRANSITION_ORDINAL || '_' || TRANSITION_CHARGE, ';') AS aggr_Fragment_Annotation
                FROM data
                WHERE TRANSITION_ID IS NOT NULL
                AND (NOT TRANSITION_DECOY OR TRANSITION_DECOY IS NULL)
                AND (SCORE_TRANSITION_PEP IS NULL OR SCORE_TRANSITION_PEP < {self.config.max_transition_pep})
                GROUP BY FEATURE_ID
            """
        else:
            query = """
                SELECT 
                    FEATURE_ID AS id,
                    STRING_AGG(CAST(FEATURE_TRANSITION_AREA_INTENSITY AS VARCHAR), ';') AS aggr_Peak_Area,
                    STRING_AGG(CAST(FEATURE_TRANSITION_APEX_INTENSITY AS VARCHAR), ';') AS aggr_Peak_Apex,
                    STRING_AGG(TRANSITION_ID || '_' || TRANSITION_TYPE || TRANSITION_ORDINAL || '_' || TRANSITION_CHARGE, ';') AS aggr_Fragment_Annotation
                FROM data
                WHERE TRANSITION_ID IS NOT NULL
                AND (NOT TRANSITION_DECOY OR TRANSITION_DECOY IS NULL)
                GROUP BY FEATURE_ID
            """

        transition_data = con.execute(query).fetchdf()
        return pd.merge(data, transition_data, on="id", how="left")

    def _add_protein_data(self, data, con) -> pd.DataFrame:
        """
        Add protein identifier data.
        """
        query = """
            SELECT 
                PEPTIDE_ID AS id_peptide,
                STRING_AGG(PROTEIN_ACCESSION, ';') AS ProteinName
            FROM data
            WHERE PROTEIN_ID IS NOT NULL
            AND PROTEIN_ACCESSION IS NOT NULL
            GROUP BY PEPTIDE_ID
        """
        protein_data = con.execute(query).fetchdf()
        return pd.merge(data, protein_data, on="id_peptide", how="inner")

    def _add_peptide_data(self, data, con) -> pd.DataFrame:
        """
        Add peptide-level error rate data.
        """
        if not any(col.startswith("SCORE_PEPTIDE_") for col in self._columns):
            return data

        # Initialize with empty DataFrame to ensure consistent structure
        peptide_data = pd.DataFrame()
        only_global_present = True

        # Run-specific peptide scores
        if any(col.startswith("SCORE_PEPTIDE_RUN_SPECIFIC_") for col in self._columns):
            logger.debug("Adding run-specific peptide scores.")
            query = """
                SELECT 
                    RUN_ID AS id_run,
                    PEPTIDE_ID AS id_peptide,
                    SCORE_PEPTIDE_RUN_SPECIFIC_Q_VALUE AS m_score_peptide_run_specific
                FROM data
                WHERE PROTEIN_ID IS NOT NULL
            """
            run_data = con.execute(query).fetchdf()
            if not run_data.empty:
                only_global_present = False
                peptide_data = run_data
                logger.trace(f"Run-specific peptide data shape: {run_data.shape}")

        # Experiment-wide peptide scores
        if any(
            col.startswith("SCORE_PEPTIDE_EXPERIMENT_WIDE_") for col in self._columns
        ):
            logger.debug("Adding experiment-wide peptide scores.")
            query = """
                SELECT 
                    RUN_ID AS id_run,
                    PEPTIDE_ID AS id_peptide,
                    SCORE_PEPTIDE_EXPERIMENT_WIDE_Q_VALUE AS m_score_peptide_experiment_wide
                FROM data
                WHERE PROTEIN_ID IS NOT NULL
            """
            exp_data = con.execute(query).fetchdf()
            if not exp_data.empty:
                logger.trace(f"Experiment-wide peptide data shape: {exp_data.shape}")
                only_global_present = False
                if peptide_data.empty:
                    peptide_data = exp_data
                else:
                    peptide_data = pd.merge(
                        peptide_data, exp_data, on=["id_run", "id_peptide"]
                    )

        # Global peptide scores
        if any(col.startswith("SCORE_PEPTIDE_GLOBAL_") for col in self._columns):
            logger.debug("Adding global peptide scores.")
            query = f"""
                SELECT 
                    PEPTIDE_ID AS id_peptide,
                    SCORE_PEPTIDE_GLOBAL_Q_VALUE AS m_score_peptide_global
                FROM data
                WHERE PROTEIN_ID IS NOT NULL
                AND SCORE_PEPTIDE_GLOBAL_Q_VALUE < {self.config.max_global_peptide_qvalue}
            """
            global_data = con.execute(query).fetchdf()
            if not global_data.empty:
                # We need to drop duplicates to avoid an explosion on the merge
                global_data = global_data.drop_duplicates(
                    subset="id_peptide", keep="first"
                )
                logger.trace(f"Global peptide data shape: {global_data.shape}")
                if peptide_data.empty:
                    peptide_data = global_data
                else:
                    peptide_data = pd.merge(peptide_data, global_data, on="id_peptide")

        if not peptide_data.empty:
            key_cols = (
                ["id_run", "id_peptide"] if not only_global_present else ["id_peptide"]
            )
            peptide_data = peptide_data.drop_duplicates(subset=key_cols, keep="first")
            logger.debug("Merging peptide data into main DataFrame.")
            logger.trace(f"Data shape before merge: {data.shape}")
            logger.trace(f"Peptide data shape: {peptide_data.shape}")
            logger.trace(f"Data head:\n{data.head()}")
            logger.trace(f"Peptide data head:\n{peptide_data.head()}")
            if only_global_present:
                data = pd.merge(data, peptide_data, on=["id_peptide"], how="left")
            else:
                data = pd.merge(
                    data, peptide_data, on=["id_run", "id_peptide"], how="left"
                )
            logger.trace(f"Merged data shape: {data.shape}")

        return data

    def _add_protein_error_data(self, data, con) -> pd.DataFrame:
        """
        Add protein-level error rate data.
        """
        if not any(col.startswith("SCORE_PROTEIN_") for col in self._columns):
            return data

        # Initialize with empty DataFrame to ensure consistent structure
        protein_data = pd.DataFrame()
        only_global_present = True

        # Run-specific protein scores
        if any(col.startswith("SCORE_PROTEIN_RUN_SPECIFIC_") for col in self._columns):
            logger.debug("Adding run-specific protein scores.")
            query = """
                SELECT 
                    d1.RUN_ID AS id_run,
                    d1.PEPTIDE_ID AS id_peptide,
                    MIN(d2.SCORE_PROTEIN_RUN_SPECIFIC_Q_VALUE) AS m_score_protein_run_specific
                FROM data d1
                JOIN data d2 ON d1.PROTEIN_ID = d2.PROTEIN_ID AND d1.RUN_ID = d2.RUN_ID
                WHERE d1.PROTEIN_ID IS NOT NULL
                GROUP BY d1.RUN_ID, d1.PEPTIDE_ID
            """
            run_data = con.execute(query).fetchdf()
            if not run_data.empty:
                protein_data = run_data
                logger.trace(f"Run-specific protein data shape: {run_data.shape}")

        # Experiment-wide protein scores
        if any(
            col.startswith("SCORE_PROTEIN_EXPERIMENT_WIDE_") for col in self._columns
        ):
            logger.debug("Adding experiment-wide protein scores.")
            query = """
                SELECT 
                    d1.RUN_ID AS id_run,
                    d1.PEPTIDE_ID AS id_peptide,
                    MIN(d2.SCORE_PROTEIN_EXPERIMENT_WIDE_Q_VALUE) AS m_score_protein_experiment_wide
                FROM data d1
                JOIN data d2 ON d1.PROTEIN_ID = d2.PROTEIN_ID AND d1.RUN_ID = d2.RUN_ID
                WHERE d1.PROTEIN_ID IS NOT NULL
                GROUP BY d1.RUN_ID, d1.PEPTIDE_ID
            """
            exp_data = con.execute(query).fetchdf()
            if not exp_data.empty:
                logger.trace(f"Experiment-wide protein data shape: {exp_data.shape}")
                only_global_present = False
                if protein_data.empty:
                    protein_data = exp_data
                else:
                    protein_data = pd.merge(
                        protein_data, exp_data, on=["id_run", "id_peptide"]
                    )

        # Global protein scores
        if any(col.startswith("SCORE_PROTEIN_GLOBAL_") for col in self._columns):
            logger.debug("Adding global protein scores.")
            query = f"""
                SELECT 
                    d1.PEPTIDE_ID AS id_peptide,
                    MIN(d2.SCORE_PROTEIN_GLOBAL_Q_VALUE) AS m_score_protein_global
                FROM data d1
                JOIN data d2 ON d1.PROTEIN_ID = d2.PROTEIN_ID
                WHERE d1.PROTEIN_ID IS NOT NULL
                AND d2.SCORE_PROTEIN_GLOBAL_Q_VALUE < {self.config.max_global_protein_qvalue}
                GROUP BY d1.PEPTIDE_ID
            """
            global_data = con.execute(query).fetchdf()
            if not global_data.empty:
                # We need to drop duplicates to avoid an explosion on the merge
                global_data = global_data.drop_duplicates(
                    subset="id_peptide", keep="first"
                )
                logger.trace(f"Global protein data shape: {global_data.shape}")
                if protein_data.empty:
                    protein_data = global_data
                else:
                    protein_data = pd.merge(protein_data, global_data, on="id_peptide")

        if not protein_data.empty:
            key_cols = (
                ["id_run", "id_peptide"] if not only_global_present else ["id_peptide"]
            )
            protein_data = protein_data.drop_duplicates(subset=key_cols, keep="first")
            logger.trace(f"Data shape before merge: {data.shape}")
            logger.trace(f"Protein data shape: {protein_data.shape}")
            logger.trace(f"Data head:\n{data.head()}")
            logger.trace(f"Protein data head:\n{protein_data.head()}")
            if only_global_present:
                data = pd.merge(data, protein_data, on=["id_peptide"], how="left")
            else:
                data = pd.merge(
                    data, protein_data, on=["id_run", "id_peptide"], how="left"
                )
            logger.trace(f"Merged data shape: {data.shape}")

        return data

    def _get_ms1_score_info(self) -> tuple[str, str]:
        """
        Get MS1 score information if available.
        """
        if self._has_ms1_scores:
            return "SCORE_MS1_PEP", ""
        return "NULL", ""

    def _build_feature_vars_sql(self) -> str:
        """
        Build SQL fragment for feature variables.
        """
        feature_vars = []
        for col in self._columns:
            if col.startswith("FEATURE_MS1_VAR_"):
                var_name = col[15:].lower()
                feature_vars.append(f"{col} AS var_ms1_{var_name}")
            elif col.startswith("FEATURE_MS2_VAR_"):
                var_name = col[15:].lower()
                feature_vars.append(f"{col} AS var_ms2_{var_name}")

        return ", " + ", ".join(feature_vars) if feature_vars else ""


class ParquetWriter(BaseParquetWriter):
    """
    Class for writing OpenSWATH results to various formats.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)
