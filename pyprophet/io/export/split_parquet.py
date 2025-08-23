import os
import glob
import pandas as pd
import duckdb
from loguru import logger

from ..util import get_parquet_column_names
from .._base import BaseSplitParquetReader, BaseSplitParquetWriter
from ..._config import ExportIOConfig


class SplitParquetReader(BaseSplitParquetReader):
    """
    Class for reading and processing data from an OpenSWATH workflow parquet split based file.
    Extended to support exporting functionality.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)

        # Check columns from the first precursor file
        precursor_files = self._get_precursor_files()
        transition_files = self._get_transition_files()
        if precursor_files:
            self._columns = get_parquet_column_names(precursor_files[0])
        else:
            raise ValueError("No precursor_features.parquet files found")

        if transition_files:
            transition_columns = get_parquet_column_names(transition_files[0])
            self._columns.extend(transition_columns)
        else:
            raise ValueError("No transition_features.parquet files found")

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

    def _get_precursor_files(self):
        """Helper to get precursor files based on structure"""
        if os.path.isdir(self.infile) and any(
            f.endswith(".oswpq") for f in os.listdir(self.infile)
        ):
            return glob.glob(
                os.path.join(self.infile, "*.oswpq", "precursors_features.parquet")
            )
        return [os.path.join(self.infile, "precursors_features.parquet")]

    def _get_transition_files(self):
        """Helper to get transition files based on structure"""
        if os.path.isdir(self.infile) and any(
            f.endswith(".oswpq") for f in os.listdir(self.infile)
        ):
            return glob.glob(
                os.path.join(self.infile, "*.oswpq", "transition_features.parquet")
            )
        return [os.path.join(self.infile, "transition_features.parquet")]

    def read(self) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            self._init_duckdb_views(con)

            if self.config.export_format == "library":
                if self._is_unscored_file():
                    descr= "Files must be scored for library generation."
                    logger.exception(descr)
                    raise ValueError(descr)
                if not self._has_peptide_protein_global_scores():
                    descr= "Files must have peptide and protein level global scores for library generation."
                    logger.exception(descr)
                    raise ValueError(descr)
                logger.info("Reading standard OpenSWATH data for library from split Parquet files.")
                return self._read_library_data(con)

            if self._is_unscored_file():
                logger.info("Reading unscored data from split Parquet files.")
                return self._read_unscored_data(con)

            if self._has_ipf and self.config.ipf == "peptidoform":
                logger.info("Reading peptidoform IPF data from split Parquet files.")
                data = self._read_peptidoform_data(con)
            elif self._has_ipf and self.config.ipf == "augmented":
                logger.info("Reading augmented data with IPF from split Parquet files.")
                data = self._read_augmented_data(con)
            else:
                logger.info("Reading standard OpenSWATH data from split Parquet files.")
                data = self._read_standard_data(con)

                return self._augment_data(data, con)
        finally:
            con.close()
    
    def _has_peptide_protein_global_scores(self) -> bool:
        """
        Check if files contain peptide and protein global scores
        """
        has_peptide = any(col.startswith("SCORE_PEPTIDE_GLOBAL") for col in self._columns)
        has_protein = any(col.startswith("SCORE_PROTEIN_GLOBAL") for col in self._columns)
        return has_peptide and has_protein

    def _is_unscored_file(self) -> bool:
        """
        Check if the files are unscored by verifying the presence of the 'SCORE_' columns.
        """
        return all(not col.startswith("SCORE_") for col in self._columns)

    def _read_unscored_data(self, con) -> pd.DataFrame:
        """
        Read unscored data from split Parquet files.
        """
        feature_vars_sql = self._build_feature_vars_sql()

        query = f"""
            SELECT
                p.RUN_ID AS id_run,
                p.PEPTIDE_ID AS id_peptide,
                p.PRECURSOR_ID AS transition_group_id,
                p.PRECURSOR_DECOY AS decoy,
                p.RUN_ID AS run_id,
                p.FILENAME AS filename,
                p.EXP_RT AS RT,
                p.EXP_RT - p.DELTA_RT AS assay_rt,
                p.DELTA_RT AS delta_rt,
                p.PRECURSOR_LIBRARY_RT AS assay_RT,
                p.NORM_RT - p.PRECURSOR_LIBRARY_RT AS delta_RT,
                p.FEATURE_ID AS id,
                p.PRECURSOR_CHARGE AS Charge,
                p.PRECURSOR_MZ AS mz,
                p.FEATURE_MS2_AREA_INTENSITY AS Intensity,
                p.FEATURE_MS1_AREA_INTENSITY AS aggr_prec_Peak_Area,
                p.FEATURE_MS1_APEX_INTENSITY AS aggr_prec_Peak_Apex,
                p.LEFT_WIDTH AS leftWidth,
                p.RIGHT_WIDTH AS rightWidth
                {feature_vars_sql}
            FROM precursors p
            WHERE p.PROTEIN_ID IS NOT NULL  -- Filter to precursor rows
            ORDER BY transition_group_id
        """
        return con.execute(query).fetchdf()

    def _read_peptidoform_data(self, con) -> pd.DataFrame:
        """
        Read data with peptidoform IPF information from split files.
        """
        score_ms1_pep, _link_ms1 = self._get_ms1_score_info()

        query = f"""
            SELECT
                p.RUN_ID AS id_run,
                p.PEPTIDE_ID AS id_peptide,
                (p.MODIFIED_SEQUENCE || '_' || p.PRECURSOR_ID) AS transition_group_id,
                p.PRECURSOR_DECOY AS decoy,
                p.RUN_ID AS run_id,
                p.FILENAME AS filename,
                p.EXP_RT AS RT,
                p.EXP_RT - p.DELTA_RT AS assay_rt,
                p.DELTA_RT AS delta_rt,
                p.NORM_RT AS iRT,
                p.PRECURSOR_LIBRARY_RT AS assay_iRT,
                p.NORM_RT - p.PRECURSOR_LIBRARY_RT AS delta_iRT,
                p.FEATURE_ID AS id,
                p.UNMODIFIED_SEQUENCE AS Sequence,
                p.MODIFIED_SEQUENCE AS FullPeptideName,
                p.PRECURSOR_CHARGE AS Charge,
                p.PRECURSOR_MZ AS mz,
                p.FEATURE_MS2_AREA_INTENSITY AS Intensity,
                p.FEATURE_MS1_AREA_INTENSITY AS aggr_prec_Peak_Area,
                p.FEATURE_MS1_APEX_INTENSITY AS aggr_prec_Peak_Apex,
                p.LEFT_WIDTH AS leftWidth,
                p.RIGHT_WIDTH AS rightWidth,
                {score_ms1_pep} AS ms1_pep,
                p.SCORE_MS2_PEP AS ms2_pep,
                p.SCORE_IPF_PRECURSOR_PEAKGROUP_PEP AS precursor_pep,
                p.SCORE_IPF_PEP AS ipf_pep,
                p.SCORE_MS2_PEAK_GROUP_RANK AS peak_group_rank,
                p.SCORE_MS2_SCORE AS d_score,
                p.SCORE_MS2_Q_VALUE AS ms2_m_score,
                p.SCORE_IPF_QVALUE AS m_score
            FROM precursors p
            WHERE p.PROTEIN_ID IS NOT NULL
            AND p.SCORE_MS2_Q_VALUE < {self.config.max_rs_peakgroup_qvalue} 
            AND p.SCORE_IPF_PEP < {self.config.ipf_max_peptidoform_pep}
            ORDER BY transition_group_id, peak_group_rank
        """
        return con.execute(query).fetchdf()

    def _read_augmented_data(self, con) -> pd.DataFrame:
        """
        Read standard data augmented with IPF information from split files.
        """
        score_ms1_pep, _link_ms1 = self._get_ms1_score_info()

        # First get main data
        query = f"""
            SELECT
                p.RUN_ID AS id_run,
                p.PEPTIDE_ID AS id_peptide,
                p.PRECURSOR_ID AS transition_group_id,
                p.PRECURSOR_DECOY AS decoy,
                p.RUN_ID AS run_id,
                p.FILENAME AS filename,
                p.EXP_RT AS RT,
                p.EXP_RT - p.DELTA_RT AS assay_rt,
                p.DELTA_RT AS delta_rt,
                p.NORM_RT AS iRT,
                p.PRECURSOR_LIBRARY_RT AS assay_iRT,
                p.NORM_RT - p.PRECURSOR_LIBRARY_RT AS delta_iRT,
                p.FEATURE_ID AS id,
                p.UNMODIFIED_SEQUENCE AS Sequence,
                p.MODIFIED_SEQUENCE AS FullPeptideName,
                p.PRECURSOR_CHARGE AS Charge,
                p.PRECURSOR_MZ AS mz,
                p.FEATURE_MS2_AREA_INTENSITY AS Intensity,
                p.FEATURE_MS1_AREA_INTENSITY AS aggr_prec_Peak_Area,
                p.FEATURE_MS1_APEX_INTENSITY AS aggr_prec_Peak_Apex,
                p.LEFT_WIDTH AS leftWidth,
                p.RIGHT_WIDTH AS rightWidth,
                p.SCORE_MS2_PEAK_GROUP_RANK AS peak_group_rank,
                p.SCORE_MS2_SCORE AS d_score,
                p.SCORE_MS2_Q_VALUE AS m_score,
                {score_ms1_pep} AS ms1_pep,
                p.SCORE_MS2_PEP AS ms2_pep
            FROM precursors p
            WHERE p.PROTEIN_ID IS NOT NULL
            AND p.SCORE_MS2_Q_VALUE < {self.config.max_rs_peakgroup_qvalue}
            ORDER BY transition_group_id, p.peak_group_rank
        """
        data = con.execute(query).fetchdf()

        # Augment with IPF data
        ipf_query = f"""
            SELECT 
                p.FEATURE_ID AS id,
                p.MODIFIED_SEQUENCE AS ipf_FullUniModPeptideName,
                p.SCORE_IPF_PRECURSOR_PEAKGROUP_PEP AS ipf_precursor_peakgroup_pep,
                p.SCORE_IPF_PEP AS ipf_peptidoform_pep,
                p.SCORE_IPF_QVALUE AS ipf_peptidoform_m_score
            FROM precursors p
            WHERE p.SCORE_IPF_PEP < {self.config.ipf_max_peptidoform_pep}
            AND p.PROTEIN_ID IS NOT NULL
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

    def _read_library_data(self, con) -> pd.DataFrame:
        """
        Read data specifically for precursors for library generation. This does not include all output in standard output
        """
        if self.config.rt_calibration:
            rt_col = "p.EXP_RT"
        else:
            rt_col = "p.PRECURSOR_LIBRARY_RT"

        if self.config.im_calibration:
            im_col = "p.EXP_IM"
        else:
            im_col = "p.PRECURSOR_LIBRARY_DRIFT_TIME"

        if self.config.intensity_calibration:
            intensity_col = 't.FEATURE_TRANSITION_AREA_INTENSITY'
        else:
            intensity_col = 't.TRANSITION_LIBRARY_INTENSITY'
        
        if self.config.keep_decoys:
            decoy_query = ""
        else:
            decoy_query ="p.PRECURSOR_DECOY is false and t.TRANSITION_DECOY is false and" 

        query = f"""
            SELECT
                {rt_col} as NormalizedRetentionTime,
                {im_col} as PrecursorIonMobility,
                {intensity_col} as LibraryIntensity,
                p.SCORE_MS2_Q_VALUE as Q_Value,
                p.UNMODIFIED_SEQUENCE AS PeptideSequence,
                p.MODIFIED_SEQUENCE AS ModifiedPeptideSequence,
                p.PRECURSOR_CHARGE AS PrecursorCharge,
                p.FEATURE_MS2_AREA_INTENSITY AS Intensity,
                p.RUN_ID AS RunId,
                (p.MODIFIED_SEQUENCE || '_' || CAST(p.PRECURSOR_CHARGE AS VARCHAR)) AS Precursor,
                p.PRECURSOR_MZ AS PrecursorMz,
                STRING_AGG(p.PROTEIN_ACCESSION, ';') AS ProteinName,
                t.ANNOTATION as Annotation,
                t.PRODUCT_MZ as ProductMz,
                t.TRANSITION_CHARGE as FragmentCharge,
                t.TRANSITION_TYPE as FragmentType,
                t.TRANSITION_ORDINAL as FragmentSeriesNumber,
                t.TRANSITION_ID as TransitionId,
                p.PRECURSOR_DECOY as Decoy
            FROM precursors p
            INNER JOIN transition t ON p.FEATURE_ID = t.FEATURE_ID
            WHERE {decoy_query} 
                  p.SCORE_MS2_Q_VALUE < {self.config.max_rs_peakgroup_qvalue} and
                  p.SCORE_PROTEIN_GLOBAL_Q_VALUE < {self.config.max_global_protein_qvalue} and
                  p.SCORE_PEPTIDE_GLOBAL_Q_VALUE < {self.config.max_global_peptide_qvalue} and
                  p.SCORE_MS2_PEAK_GROUP_RANK = 1

            GROUP BY {rt_col}, {im_col}, {intensity_col}, p.SCORE_MS2_Q_VALUE,
                     p.UNMODIFIED_SEQUENCE, p.MODIFIED_SEQUENCE, p.PRECURSOR_CHARGE,
                     p.PRECURSOR_MZ, p.FEATURE_ID, t.ANNOTATION, t.PRODUCT_MZ,
                     t.TRANSITION_CHARGE, t.TRANSITION_TYPE, t.TRANSITION_ORDINAL, t.TRANSITION_ID, p.PRECURSOR_DECOY, p.RUN_ID
        """
        return con.execute(query).fetchdf()
    
    def _read_standard_data(self, con) -> pd.DataFrame:
        """
        Read standard OpenSWATH data without IPF from split files.
        """
        query = f"""
            SELECT
                p.RUN_ID AS id_run,
                p.PEPTIDE_ID AS id_peptide,
                p.PRECURSOR_ID AS transition_group_id,
                p.PRECURSOR_DECOY AS decoy,
                p.RUN_ID AS run_id,
                p.FILENAME AS filename,
                p.EXP_RT AS RT,
                p.EXP_RT - p.DELTA_RT AS assay_rt,
                p.DELTA_RT AS delta_rt,
                p.NORM_RT AS iRT,
                p.PRECURSOR_LIBRARY_RT AS assay_iRT,
                p.NORM_RT - p.PRECURSOR_LIBRARY_RT AS delta_iRT,
                p.FEATURE_ID AS id,
                p.UNMODIFIED_SEQUENCE AS Sequence,
                p.MODIFIED_SEQUENCE AS FullPeptideName,
                p.PRECURSOR_CHARGE AS Charge,
                p.PRECURSOR_MZ AS mz,
                p.FEATURE_MS2_AREA_INTENSITY AS Intensity,
                p.FEATURE_MS1_AREA_INTENSITY AS aggr_prec_Peak_Area,
                p.FEATURE_MS1_APEX_INTENSITY AS aggr_prec_Peak_Apex,
                p.LEFT_WIDTH AS leftWidth,
                p.RIGHT_WIDTH AS rightWidth,
                p.SCORE_MS2_PEAK_GROUP_RANK AS peak_group_rank,
                p.SCORE_MS2_SCORE AS d_score,
                p.SCORE_MS2_Q_VALUE AS m_score
            FROM precursors p
            WHERE p.PROTEIN_ID IS NOT NULL
            AND p.SCORE_MS2_Q_VALUE < {self.config.max_rs_peakgroup_qvalue}
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
        Add transition-level quantification data from split files.
        """
        if self._has_transition_scores:
            query = f"""
                SELECT 
                    t.FEATURE_ID AS id,
                    STRING_AGG(CAST(t.FEATURE_TRANSITION_AREA_INTENSITY AS VARCHAR), ';') AS aggr_Peak_Area,
                    STRING_AGG(CAST(t.FEATURE_TRANSITION_APEX_INTENSITY AS VARCHAR), ';') AS aggr_Peak_Apex,
                    STRING_AGG(t.TRANSITION_ID || '_' || t.TRANSITION_TYPE || t.TRANSITION_ORDINAL || '_' || t.TRANSITION_CHARGE, ';') AS aggr_Fragment_Annotation
                FROM transition t
                WHERE t.TRANSITION_ID IS NOT NULL
                AND (NOT t.TRANSITION_DECOY OR t.TRANSITION_DECOY IS NULL)
                AND (t.SCORE_TRANSITION_PEP IS NULL OR t.SCORE_TRANSITION_PEP < {self.config.max_transition_pep})
                GROUP BY t.FEATURE_ID
            """
        else:
            query = """
                SELECT 
                    t.FEATURE_ID AS id,
                    STRING_AGG(CAST(t.FEATURE_TRANSITION_AREA_INTENSITY AS VARCHAR), ';') AS aggr_Peak_Area,
                    STRING_AGG(CAST(t.FEATURE_TRANSITION_APEX_INTENSITY AS VARCHAR), ';') AS aggr_Peak_Apex,
                    STRING_AGG(t.TRANSITION_ID || '_' || t.TRANSITION_TYPE || t.TRANSITION_ORDINAL || '_' || t.TRANSITION_CHARGE, ';') AS aggr_Fragment_Annotation
                FROM transition t
                WHERE t.TRANSITION_ID IS NOT NULL
                AND (NOT t.TRANSITION_DECOY OR t.TRANSITION_DECOY IS NULL)
                GROUP BY t.FEATURE_ID
            """

        transition_data = con.execute(query).fetchdf()
        return pd.merge(data, transition_data, on="id", how="left")

    def _add_protein_data(self, data, con) -> pd.DataFrame:
        """
        Add protein identifier data from split files.
        """
        query = """
            SELECT 
                p.PEPTIDE_ID AS id_peptide,
                STRING_AGG(p.PROTEIN_ACCESSION, ';') AS ProteinName
            FROM precursors p
            WHERE p.PROTEIN_ID IS NOT NULL
            AND p.PROTEIN_ACCESSION IS NOT NULL
            GROUP BY p.PEPTIDE_ID
        """
        protein_data = con.execute(query).fetchdf()
        return pd.merge(data, protein_data, on="id_peptide", how="inner")

    def _add_peptide_data(self, data, con) -> pd.DataFrame:
        """
        Add peptide-level error rate data from split files.
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
                    p.RUN_ID AS id_run,
                    p.PEPTIDE_ID AS id_peptide,
                    p.SCORE_PEPTIDE_RUN_SPECIFIC_Q_VALUE AS m_score_peptide_run_specific
                FROM precursors p
                WHERE p.PROTEIN_ID IS NOT NULL
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
                    p.RUN_ID AS id_run,
                    p.PEPTIDE_ID AS id_peptide,
                    p.SCORE_PEPTIDE_EXPERIMENT_WIDE_Q_VALUE AS m_score_peptide_experiment_wide
                FROM precursors p
                WHERE p.PROTEIN_ID IS NOT NULL
            """
            exp_data = con.execute(query).fetchdf()
            if not exp_data.empty:
                logger.trace(f"Experiment-wide peptide data shape: {exp_data.shape}")
                only_global_present = False
                if peptide_data.empty:
                    peptide_data = exp_data
                else:
                    peptide_data = pd.merge(
                        peptide_data, exp_data, on=["id_run", "id_peptide"], how="left"
                    )

        # Global peptide scores
        if any(col.startswith("SCORE_PEPTIDE_GLOBAL_") for col in self._columns):
            logger.debug("Adding global peptide scores.")
            query = f"""
                SELECT 
                    p.PEPTIDE_ID AS id_peptide,
                    p.SCORE_PEPTIDE_GLOBAL_Q_VALUE AS m_score_peptide_global
                FROM precursors p
                WHERE p.PROTEIN_ID IS NOT NULL
                AND p.SCORE_PEPTIDE_GLOBAL_Q_VALUE < {self.config.max_global_peptide_qvalue}
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
                    peptide_data = pd.merge(
                        peptide_data, global_data, on="id_peptide", how="left"
                    )

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
        Add protein-level error rate data from split files.
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
                    p1.RUN_ID AS id_run,
                    p1.PEPTIDE_ID AS id_peptide,
                    MIN(p1.SCORE_PROTEIN_RUN_SPECIFIC_Q_VALUE) AS m_score_protein_run_specific
                FROM precursors p1
                -- JOIN precursors p2 ON p1.PROTEIN_ID = p2.PROTEIN_ID AND p1.RUN_ID = p2.RUN_ID
                WHERE p1.PROTEIN_ID IS NOT NULL
                GROUP BY p1.RUN_ID, p1.PEPTIDE_ID
            """
            run_data = con.execute(query).fetchdf()
            if not run_data.empty:
                only_global_present = False
                protein_data = run_data
                logger.trace(f"Run-specific protein data shape: {run_data.shape}")

        # Experiment-wide protein scores
        if any(
            col.startswith("SCORE_PROTEIN_EXPERIMENT_WIDE_") for col in self._columns
        ):
            logger.debug("Adding experiment-wide protein scores.")
            query = """
                SELECT 
                    p1.RUN_ID AS id_run,
                    p1.PEPTIDE_ID AS id_peptide,
                    MIN(p1.SCORE_PROTEIN_EXPERIMENT_WIDE_Q_VALUE) AS m_score_protein_experiment_wide
                FROM precursors p1
                -- JOIN precursors p2 ON p1.PROTEIN_ID = p2.PROTEIN_ID AND p1.RUN_ID = p2.RUN_ID
                WHERE p1.PROTEIN_ID IS NOT NULL
                GROUP BY p1.RUN_ID, p1.PEPTIDE_ID
            """
            exp_data = con.execute(query).fetchdf()
            if not exp_data.empty:
                logger.trace(f"Experiment-wide protein data shape: {exp_data.shape}")
                only_global_present = False
                if protein_data.empty:
                    protein_data = exp_data
                else:
                    protein_data = pd.merge(
                        protein_data, exp_data, on=["id_run", "id_peptide"], how="left"
                    )

        # Global protein scores
        if any(col.startswith("SCORE_PROTEIN_GLOBAL_") for col in self._columns):
            logger.debug("Adding global protein scores.")
            query = f"""
                SELECT 
                    p1.PEPTIDE_ID AS id_peptide,
                    MIN(p1.SCORE_PROTEIN_GLOBAL_Q_VALUE) AS m_score_protein_global
                FROM precursors p1
                -- JOIN precursors p2 ON p1.PROTEIN_ID = p2.PROTEIN_ID
                WHERE p1.PROTEIN_ID IS NOT NULL
                AND p1.SCORE_PROTEIN_GLOBAL_Q_VALUE < {self.config.max_global_protein_qvalue}
                GROUP BY p1.PEPTIDE_ID
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
                    protein_data = pd.merge(
                        protein_data, global_data, on="id_peptide", how="left"
                    )

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
            return "p.SCORE_MS1_PEP", ""
        return "NULL", ""

    def _build_feature_vars_sql(self) -> str:
        """
        Build SQL fragment for feature variables.
        """
        feature_vars = []
        for col in self._columns:
            if col.startswith("FEATURE_MS1_VAR_"):
                var_name = col[15:].lower()
                feature_vars.append(f"p.{col} AS var_ms1_{var_name}")
            elif col.startswith("FEATURE_MS2_VAR_"):
                var_name = col[15:].lower()
                feature_vars.append(f"p.{col} AS var_ms2_{var_name}")

        return ", " + ", ".join(feature_vars) if feature_vars else ""


class SplitParquetWriter(BaseSplitParquetWriter):
    """
    Class for writing OpenSWATH results to various formats.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)
