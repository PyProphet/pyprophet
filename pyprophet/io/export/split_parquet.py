import os
import glob
import pandas as pd
import duckdb
from loguru import logger

from ..util import get_parquet_column_names, _ensure_pyarrow
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

        # Check for alignment file
        self._has_alignment = self._check_alignment_file_exists()

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
                    descr = "Files must be scored for library generation."
                    logger.exception(descr)
                    raise ValueError(descr)
                if not self._has_peptide_protein_global_scores():
                    descr = "Files must have peptide and protein level global scores for library generation."
                    logger.exception(descr)
                    raise ValueError(descr)
                logger.info(
                    "Reading standard OpenSWATH data for library from split Parquet files."
                )
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
        has_peptide = any(
            col.startswith("SCORE_PEPTIDE_GLOBAL") for col in self._columns
        )
        has_protein = any(
            col.startswith("SCORE_PROTEIN_GLOBAL") for col in self._columns
        )
        return has_peptide and has_protein

    def _is_unscored_file(self) -> bool:
        """
        Check if the files are unscored by verifying the presence of the 'SCORE_' columns.
        """
        return all(not col.startswith("SCORE_") for col in self._columns)

    def _check_alignment_file_exists(self) -> bool:
        """
        Check if alignment parquet file exists for split parquet format.

        For split parquet, alignment file is at the parent directory level:
        - infile is a directory containing *.oswpq subdirectories
        - alignment file is at infile/feature_alignment.parquet
        """
        import os

        alignment_file = None
        if os.path.isdir(self.infile):
            # Split parquet format: alignment file is in the parent directory
            alignment_file = os.path.join(self.infile, "feature_alignment.parquet")

        if alignment_file and os.path.exists(alignment_file):
            logger.debug(f"Alignment file found: {alignment_file}")
            return True
        return False

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
            intensity_col = "t.FEATURE_TRANSITION_AREA_INTENSITY"
        else:
            intensity_col = "t.TRANSITION_LIBRARY_INTENSITY"

        if self.config.keep_decoys:
            decoy_query = ""
        else:
            decoy_query = (
                "p.PRECURSOR_DECOY is false and t.TRANSITION_DECOY is false and"
            )

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
                     t.TRANSITION_CHARGE, t.TRANSITION_TYPE, t.TRANSITION_ORDINAL, t.TRANSITION_ID, p.PRECURSOR_DECOY, p.RUN_ID, p.FEATURE_MS2_AREA_INTENSITY
        """
        return con.execute(query).fetchdf()

    def _read_standard_data(self, con) -> pd.DataFrame:
        """
        Read standard OpenSWATH data without IPF from split files, optionally including aligned features.
        """
        # Check if we should attempt alignment integration
        use_alignment = self.config.use_alignment and self._has_alignment

        # First, get features that pass MS2 QVALUE threshold
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
                p.SCORE_MS2_PEP AS pep
            FROM precursors p
            WHERE p.PROTEIN_ID IS NOT NULL
            AND p.SCORE_MS2_Q_VALUE < {self.config.max_rs_peakgroup_qvalue}
            ORDER BY transition_group_id, peak_group_rank
        """
        data = con.execute(query).fetchdf()

        # If alignment is enabled and alignment data is present, fetch and merge aligned features
        if use_alignment:
            aligned_features = self._fetch_alignment_features(con)

            if not aligned_features.empty:
                # Get full feature data for aligned features that are NOT already in base results
                # We only want to add features that didn't pass MS2 threshold but have good alignment
                aligned_ids = aligned_features["id"].unique()
                existing_ids = data["id"].unique()
                new_aligned_ids = [
                    aid for aid in aligned_ids if aid not in existing_ids
                ]

                # First, merge alignment info into existing features (those that passed MS2)
                # Mark them with from_alignment=0
                if "alignment_pep" in aligned_features.columns:
                    # Build list of columns to merge
                    merge_cols = ["id", "alignment_pep", "alignment_qvalue"]
                    if "alignment_group_id" in aligned_features.columns:
                        merge_cols.append("alignment_group_id")
                    if "alignment_reference_feature_id" in aligned_features.columns:
                        merge_cols.append("alignment_reference_feature_id")
                    if "alignment_reference_rt" in aligned_features.columns:
                        merge_cols.append("alignment_reference_rt")

                    data = pd.merge(
                        data, aligned_features[merge_cols], on="id", how="left"
                    )
                data["from_alignment"] = 0

                # Now add features that didn't pass MS2 but have good alignment (recovered features)
                if new_aligned_ids:
                    # Fetch full data for these new aligned features from the main data view
                    # Register aligned IDs as a temp table for the query
                    aligned_ids_df = pd.DataFrame({"id": new_aligned_ids})
                    con.register("aligned_ids_temp", aligned_ids_df)

                    aligned_query = f"""
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
                        AND p.FEATURE_ID IN (SELECT id FROM aligned_ids_temp)
                    """
                    aligned_data = con.execute(aligned_query).fetchdf()

                    # Merge alignment scores and reference info into the aligned data
                    if "alignment_pep" in aligned_features.columns:
                        aligned_data = pd.merge(
                            aligned_data,
                            aligned_features[merge_cols],
                            on="id",
                            how="left",
                        )

                    # Mark as recovered through alignment
                    aligned_data["from_alignment"] = 1

                    logger.info(
                        f"Adding {len(aligned_data)} features recovered through alignment"
                    )

                    # Combine with base data
                    data = pd.concat([data, aligned_data], ignore_index=True)

                # Convert alignment_reference_feature_id to int64 to avoid scientific notation
                if "alignment_reference_feature_id" in data.columns:
                    data["alignment_reference_feature_id"] = data[
                        "alignment_reference_feature_id"
                    ].astype("Int64")
                if "alignment_group_id" in data.columns:
                    data["alignment_group_id"] = data["alignment_group_id"].astype(
                        "Int64"
                    )

                # Assign alignment_group_id to reference features
                # Create a mapping from reference feature IDs to their alignment_group_ids
                if (
                    "alignment_reference_feature_id" in data.columns
                    and "alignment_group_id" in data.columns
                ):
                    # Get all reference feature IDs and their corresponding alignment_group_ids
                    ref_mapping = data[data["alignment_reference_feature_id"].notna()][
                        ["alignment_reference_feature_id", "alignment_group_id"]
                    ].drop_duplicates()

                    # For each reference feature ID, we need to assign the alignment_group_id
                    # to the feature row where id == alignment_reference_feature_id
                    if not ref_mapping.empty:
                        # Merge the alignment_group_id for reference features
                        # First create a DataFrame mapping id -> alignment_group_id for references
                        ref_group_mapping = ref_mapping.rename(
                            columns={
                                "alignment_reference_feature_id": "id",
                                "alignment_group_id": "ref_alignment_group_id",
                            }
                        )

                        # Merge this mapping to assign alignment_group_id to reference features
                        data = pd.merge(data, ref_group_mapping, on="id", how="left")

                        # Fill in alignment_group_id for reference features (where it's currently null but ref_alignment_group_id is not)
                        mask = (
                            data["alignment_group_id"].isna()
                            & data["ref_alignment_group_id"].notna()
                        )
                        data.loc[mask, "alignment_group_id"] = data.loc[
                            mask, "ref_alignment_group_id"
                        ]

                        # Drop the temporary column
                        data = data.drop(columns=["ref_alignment_group_id"])

                        logger.debug(
                            f"Assigned alignment_group_id to {mask.sum()} reference features"
                        )

        return data

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

    def _fetch_alignment_features(self, con) -> pd.DataFrame:
        """
        Fetch aligned features with good alignment scores from alignment parquet file.

        This method checks for an alignment parquet file and retrieves features
        that have been aligned across runs and pass the alignment quality threshold.
        Only features whose reference feature passes the MS2 QVALUE threshold are included.

        Args:
            con: DuckDB connection

        Returns:
            DataFrame with aligned feature IDs that pass quality threshold
        """

        # For split parquet, alignment file is at parent directory level
        alignment_file = os.path.join(self.infile, "feature_alignment.parquet")

        if not os.path.exists(alignment_file):
            logger.debug(
                "Alignment parquet file not found, skipping alignment integration"
            )
            return pd.DataFrame()

        logger.debug(f"Loading alignment data from {alignment_file}")
        max_alignment_pep = self.config.max_alignment_pep
        max_rs_peakgroup_qvalue = self.config.max_rs_peakgroup_qvalue

        try:
            # Load alignment data
            alignment_df = pd.read_parquet(alignment_file)

            # Filter to target (non-decoy) features with good alignment scores
            # Note: DECOY column in parquet alignment file comes from LABEL in SQLite
            # where LABEL=1 (DECOY=1 in parquet) means target, not decoy
            if (
                "DECOY" in alignment_df.columns
                and "VAR_XCORR_SHAPE" in alignment_df.columns
            ):
                # Check if we have alignment scores (PEP/QVALUE) in the file
                # If not, we'll need to rely on the base MS2 scores and just use alignment to identify features
                has_alignment_scores = (
                    "SCORE_ALIGNMENT_PEP" in alignment_df.columns
                    or "SCORE_ALIGNMENT_Q_VALUE" in alignment_df.columns
                )

                if has_alignment_scores:
                    # Filter by alignment PEP threshold
                    pep_col = (
                        "SCORE_ALIGNMENT_PEP"
                        if "SCORE_ALIGNMENT_PEP" in alignment_df.columns
                        else None
                    )
                    qvalue_col = (
                        "SCORE_ALIGNMENT_Q_VALUE"
                        if "SCORE_ALIGNMENT_Q_VALUE" in alignment_df.columns
                        else None
                    )

                    if pep_col:
                        filtered_df = alignment_df[
                            (
                                alignment_df["DECOY"] == 1
                            )  # DECOY=1 means target (from LABEL=1 in SQLite)
                            & (alignment_df[pep_col] < max_alignment_pep)
                        ].copy()
                    else:
                        # Use QVALUE if SCORE_ALIGNMENT_PEP not available (less ideal but workable)
                        filtered_df = alignment_df[
                            (alignment_df["DECOY"] == 1)
                            & (alignment_df[qvalue_col] < max_alignment_pep)
                        ].copy()
                else:
                    # No alignment scores in file - just filter by target status
                    # In this case, we can't apply alignment quality threshold
                    logger.warning(
                        "Alignment file found but no PEP/QVALUE scores present. Cannot filter by alignment quality."
                    )
                    filtered_df = alignment_df[alignment_df["DECOY"] == 1].copy()

                # Now filter by reference feature MS2 QVALUE
                # Need to join with precursors data to check reference feature QVALUE
                if "REFERENCE_FEATURE_ID" in filtered_df.columns:
                    # Register filtered alignment data for SQL query
                    con.register("filtered_alignment", filtered_df)

                    # Query to get aligned features where reference passes MS2 QVALUE threshold
                    # Also compute alignment_group_id using DENSE_RANK
                    # CAST in SELECT preserves precision, but not in JOIN (for performance)
                    ref_check_query = f"""
                        SELECT 
                            DENSE_RANK() OVER (ORDER BY fa.PRECURSOR_ID, fa.ALIGNMENT_ID) AS ALIGNMENT_GROUP_ID,
                            fa.FEATURE_ID,
                            fa.PRECURSOR_ID,
                            fa.RUN_ID,
                            CAST(fa.REFERENCE_FEATURE_ID AS BIGINT) AS REFERENCE_FEATURE_ID,
                            fa.REFERENCE_RT,
                            fa.SCORE_ALIGNMENT_PEP,
                            fa.SCORE_ALIGNMENT_Q_VALUE
                        FROM filtered_alignment fa
                        INNER JOIN precursors p ON p.FEATURE_ID = fa.REFERENCE_FEATURE_ID
                        WHERE p.SCORE_MS2_Q_VALUE < {max_rs_peakgroup_qvalue}
                    """
                    filtered_df = con.execute(ref_check_query).fetchdf()

                # Rename columns to match expected format
                if "FEATURE_ID" in filtered_df.columns:
                    # Start with base columns
                    base_cols = ["FEATURE_ID", "PRECURSOR_ID", "RUN_ID"]
                    result = filtered_df[base_cols].rename(columns={"FEATURE_ID": "id"})

                    # Add alignment group ID if available
                    if "ALIGNMENT_GROUP_ID" in filtered_df.columns:
                        result["alignment_group_id"] = filtered_df[
                            "ALIGNMENT_GROUP_ID"
                        ].values

                    # Add reference feature ID and RT if available
                    # Ensure Int64 dtype to preserve precision for large IDs
                    if "REFERENCE_FEATURE_ID" in filtered_df.columns:
                        result["alignment_reference_feature_id"] = (
                            filtered_df["REFERENCE_FEATURE_ID"].astype("Int64").values
                        )
                    if "REFERENCE_RT" in filtered_df.columns:
                        result["alignment_reference_rt"] = filtered_df[
                            "REFERENCE_RT"
                        ].values

                    # Add alignment scores if available
                    if "SCORE_ALIGNMENT_PEP" in filtered_df.columns:
                        result["alignment_pep"] = filtered_df[
                            "SCORE_ALIGNMENT_PEP"
                        ].values
                    if "SCORE_ALIGNMENT_Q_VALUE" in filtered_df.columns:
                        result["alignment_qvalue"] = filtered_df[
                            "SCORE_ALIGNMENT_Q_VALUE"
                        ].values

                    # Convert alignment_group_id to int64
                    if "alignment_group_id" in result.columns:
                        result["alignment_group_id"] = result[
                            "alignment_group_id"
                        ].astype("Int64")

                    logger.info(
                        f"Found {len(result)} aligned features passing alignment PEP < {max_alignment_pep} "
                        + f"with reference features passing MS2 QVALUE < {max_rs_peakgroup_qvalue}"
                    )
                    return result
        except Exception as e:
            logger.warning(f"Could not load alignment data: {e}")

        return pd.DataFrame()

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

    def export_feature_scores(self, outfile: str, plot_callback):
        """
        Export feature scores from split Parquet directory for plotting.

        Detects if SCORE columns exist and adjusts behavior:
        - If SCORE columns exist: applies RANK==1 filtering and plots SCORE + VAR_ columns
        - If SCORE columns don't exist: plots only VAR_ columns

        Parameters
        ----------
        outfile : str
            Path to the output PDF file.
        plot_callback : callable
            Function to call for plotting each level's data.
            Signature: plot_callback(df, outfile, level, append)
        """
        # Ensure pyarrow is available
        pa, _, _ = _ensure_pyarrow()

        # Read precursor features - only necessary columns
        precursor_file = os.path.join(self.infile, "precursors_features.parquet")
        logger.info(f"Reading precursor features from: {precursor_file}")

        # First check what columns are available
        precursor_parquet = pa.parquet.ParquetFile(precursor_file)
        all_columns = precursor_parquet.schema.names

        # Check for SCORE columns
        score_cols = [col for col in all_columns if col.startswith("SCORE_")]
        has_scores = len(score_cols) > 0

        if has_scores:
            logger.info(
                "SCORE columns detected - applying RANK==1 filter and plotting SCORE + VAR_ columns"
            )
        else:
            logger.info("No SCORE columns detected - plotting only VAR_ columns")

        # Identify columns to read
        ms1_cols = [col for col in all_columns if col.startswith("FEATURE_MS1_VAR_")]
        ms2_cols = [col for col in all_columns if col.startswith("FEATURE_MS2_VAR_")]

        cols_to_read = set()

        # Add SCORE columns if they exist
        if has_scores:
            cols_to_read.update(score_cols)
            # Add RANK column for filtering
            if "SCORE_MS2_PEAK_GROUP_RANK" in all_columns:
                cols_to_read.add("SCORE_MS2_PEAK_GROUP_RANK")
            # Add ID columns for grouping
            if "RUN_ID" in all_columns:
                cols_to_read.add("RUN_ID")
            if "PRECURSOR_ID" in all_columns:
                cols_to_read.add("PRECURSOR_ID")

        if ms1_cols and "PRECURSOR_DECOY" in all_columns:
            cols_to_read.update(ms1_cols)
            cols_to_read.add("PRECURSOR_DECOY")
        if ms2_cols and "PRECURSOR_DECOY" in all_columns:
            cols_to_read.update(ms2_cols)
            cols_to_read.add("PRECURSOR_DECOY")

        if cols_to_read:
            logger.info(f"Reading {len(cols_to_read)} columns from precursor features")
            df_precursor = pd.read_parquet(precursor_file, columns=list(cols_to_read))

            # Apply RANK==1 filter if SCORE columns exist
            if has_scores and "SCORE_MS2_PEAK_GROUP_RANK" in df_precursor.columns:
                logger.info(f"Filtering to RANK==1: {len(df_precursor)} -> ", end="")
                df_precursor = df_precursor[
                    df_precursor["SCORE_MS2_PEAK_GROUP_RANK"] == 1
                ].copy()
                logger.info(f"{len(df_precursor)} rows")

            # Generate GROUP_ID if needed
            if has_scores and "GROUP_ID" not in df_precursor.columns:
                if (
                    "RUN_ID" in df_precursor.columns
                    and "PRECURSOR_ID" in df_precursor.columns
                ):
                    df_precursor["GROUP_ID"] = (
                        df_precursor["RUN_ID"].astype(str)
                        + "_"
                        + df_precursor["PRECURSOR_ID"].astype(str)
                    )

            # Process MS1 level
            if ms1_cols and "PRECURSOR_DECOY" in df_precursor.columns:
                logger.info("Processing MS1 level feature scores")
                select_cols = ms1_cols + ["PRECURSOR_DECOY"]
                # Add SCORE columns if present
                if has_scores:
                    score_ms1_cols = [col for col in score_cols if "MS1" in col.upper()]
                    select_cols.extend(score_ms1_cols)
                    if "GROUP_ID" in df_precursor.columns:
                        select_cols.append("GROUP_ID")
                ms1_df = df_precursor[select_cols].copy()
                ms1_df.rename(columns={"PRECURSOR_DECOY": "DECOY"}, inplace=True)
                plot_callback(ms1_df, outfile, "ms1", append=False)
                del ms1_df  # Free memory

            # Process MS2 level
            if ms2_cols and "PRECURSOR_DECOY" in df_precursor.columns:
                logger.info("Processing MS2 level feature scores")
                select_cols = ms2_cols + ["PRECURSOR_DECOY"]
                # Add SCORE columns if present
                if has_scores:
                    score_ms2_cols = [
                        col
                        for col in score_cols
                        if "MS2" in col.upper() or "MS1" not in col.upper()
                    ]
                    select_cols.extend(score_ms2_cols)
                    if "GROUP_ID" in df_precursor.columns:
                        select_cols.append("GROUP_ID")
                ms2_df = df_precursor[select_cols].copy()
                ms2_df.rename(columns={"PRECURSOR_DECOY": "DECOY"}, inplace=True)
                append = bool(ms1_cols)
                plot_callback(ms2_df, outfile, "ms2", append=append)
                del ms2_df  # Free memory

            del df_precursor  # Free memory

        # Read transition features if available
        transition_file = os.path.join(self.infile, "transition_features.parquet")
        if os.path.exists(transition_file):
            logger.info(f"Reading transition features from: {transition_file}")

            # Check what columns are available
            transition_parquet = pa.parquet.ParquetFile(transition_file)
            transition_all_columns = transition_parquet.schema.names
            transition_cols = [
                col
                for col in transition_all_columns
                if col.startswith("FEATURE_TRANSITION_VAR_")
            ]

            # Check for SCORE columns in transition file
            transition_score_cols = [
                col
                for col in transition_all_columns
                if col.startswith("SCORE_") and "TRANSITION" in col.upper()
            ]
            has_transition_scores = len(transition_score_cols) > 0

            if transition_cols and "TRANSITION_DECOY" in transition_all_columns:
                # Read only necessary columns
                cols_to_read = transition_cols + ["TRANSITION_DECOY"]
                if has_transition_scores:
                    cols_to_read.extend(transition_score_cols)
                    if "GROUP_ID" in transition_all_columns:
                        cols_to_read.append("GROUP_ID")

                logger.info(
                    f"Reading {len(cols_to_read)} columns from transition features"
                )
                df_transition = pd.read_parquet(transition_file, columns=cols_to_read)

                logger.info("Processing transition level feature scores")
                transition_df = df_transition.copy()
                transition_df.rename(
                    columns={"TRANSITION_DECOY": "DECOY"}, inplace=True
                )
                append = bool(ms1_cols or ms2_cols)
                plot_callback(transition_df, outfile, "transition", append=append)
                del transition_df, df_transition  # Free memory

        # Read alignment features if available
        alignment_file = os.path.join(self.infile, "feature_alignment.parquet")
        if os.path.exists(alignment_file):
            logger.info(f"Reading alignment features from: {alignment_file}")

            # Check what columns are available
            alignment_parquet = pa.parquet.ParquetFile(alignment_file)
            alignment_all_columns = alignment_parquet.schema.names
            var_cols = [col for col in alignment_all_columns if col.startswith("VAR_")]

            if var_cols and "DECOY" in alignment_all_columns:
                # Read only necessary columns
                cols_to_read = var_cols + ["DECOY"]
                logger.info(
                    f"Reading {len(cols_to_read)} columns from alignment features"
                )
                df_alignment = pd.read_parquet(alignment_file, columns=cols_to_read)

                logger.info("Processing alignment level feature scores")
                alignment_df = df_alignment[var_cols + ["DECOY"]].copy()
                append = bool(
                    ms1_cols
                    or ms2_cols
                    or (os.path.exists(transition_file) and transition_cols)
                )
                plot_callback(alignment_df, outfile, "alignment", append=append)
                del alignment_df, df_alignment  # Free memory


class SplitParquetWriter(BaseSplitParquetWriter):
    """
    Class for writing OpenSWATH results to various formats.
    """

    def __init__(self, config: ExportIOConfig):
        super().__init__(config)
