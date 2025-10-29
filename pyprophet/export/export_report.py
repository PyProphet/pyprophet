import sqlite3
import pandas as pd
import os
from loguru import logger


from .._config import ExportIOConfig
from ..report import post_scoring_report
from ..io.dispatcher import ReaderDispatcher
from ..io.util import get_parquet_column_names
from ..io.util import check_sqlite_table
from ..report import plot_scores


def export_score_plots(infile):
    """
    Export score plots from a PyProphet input file.
    """
    con = sqlite3.connect(infile)

    if check_sqlite_table(con, "SCORE_MS2"):
        outfile = infile.split(".osw")[0] + "_ms2_score_plots.pdf"
        table_ms2 = pd.read_sql_query(
            """
SELECT *,
       RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
FROM FEATURE_MS2
INNER JOIN
  (SELECT RUN_ID,
          ID,
          PRECURSOR_ID,
          EXP_RT
   FROM FEATURE) AS FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
INNER JOIN
  (SELECT ID,
          CHARGE AS VAR_PRECURSOR_CHARGE,
          DECOY
   FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN
  (SELECT PRECURSOR_ID AS ID,
          COUNT(*) AS VAR_TRANSITION_NUM_SCORE
   FROM TRANSITION_PRECURSOR_MAPPING
   INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
   WHERE DETECTING==1
   GROUP BY PRECURSOR_ID) AS VAR_TRANSITION_SCORE ON FEATURE.PRECURSOR_ID = VAR_TRANSITION_SCORE.ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
WHERE RANK == 1
ORDER BY RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
""",
            con,
        )
        plot_scores(table_ms2, outfile)

    if check_sqlite_table(con, "SCORE_MS1"):
        outfile = infile.split(".osw")[0] + "_ms1_score_plots.pdf"
        table_ms1 = pd.read_sql_query(
            """
SELECT *,
       RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
FROM FEATURE_MS1
INNER JOIN
  (SELECT RUN_ID,
          ID,
          PRECURSOR_ID,
          EXP_RT
   FROM FEATURE) AS FEATURE ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
INNER JOIN
  (SELECT ID,
          CHARGE AS VAR_PRECURSOR_CHARGE,
          DECOY
   FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
WHERE RANK == 1
ORDER BY RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
""",
            con,
        )
        plot_scores(table_ms1, outfile)

    if check_sqlite_table(con, "SCORE_TRANSITION"):
        outfile = infile.split(".osw")[0] + "_transition_score_plots.pdf"
        table_transition = pd.read_sql_query(
            """
SELECT TRANSITION.DECOY AS DECOY,
       FEATURE_TRANSITION.*,
       PRECURSOR.CHARGE AS VAR_PRECURSOR_CHARGE,
       TRANSITION.VAR_PRODUCT_CHARGE AS VAR_PRODUCT_CHARGE,
       SCORE_TRANSITION.*,
       RUN_ID || '_' || FEATURE_TRANSITION.FEATURE_ID || '_' || PRECURSOR_ID || '_' || FEATURE_TRANSITION.TRANSITION_ID AS GROUP_ID
FROM FEATURE_TRANSITION
INNER JOIN
  (SELECT RUN_ID,
          ID,
          PRECURSOR_ID,
          EXP_RT
   FROM FEATURE) AS FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN SCORE_TRANSITION ON FEATURE_TRANSITION.FEATURE_ID = SCORE_TRANSITION.FEATURE_ID
AND FEATURE_TRANSITION.TRANSITION_ID = SCORE_TRANSITION.TRANSITION_ID
INNER JOIN
  (SELECT ID,
          CHARGE AS VAR_PRODUCT_CHARGE,
          DECOY
   FROM TRANSITION) AS TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
ORDER BY RUN_ID,
         PRECURSOR.ID,
         FEATURE.EXP_RT,
         TRANSITION.ID;
""",
            con,
        )
        plot_scores(table_transition, outfile)

    con.close()


def export_scored_report(
    infile: str,
    outfile: str,
):
    """
    Export a scored report from a PyProphet input file.

    Parameters
    ----------
    infile : str
        Path to the input file (PyProphet output file).
    outfile : str
        Path to the output file.
    scoring_format : str, optional
        The format of the scoring report, either 'osw' or 'parquet'. Default is 'osw'.
    """

    config = ExportIOConfig(
        infile=infile,
        outfile=outfile,
        subsample_ratio=1.0,  # not used for export/report paths
        level="export",
        context="export_scored_report",
        # no need to set export_format for this utility
    )

    # Get the right reader for the detected file type & context.
    reader = ReaderDispatcher.get_reader(config)

    # Read once (works for OSW or Parquet via their respective readers).
    df = reader.read()

    post_scoring_report(df, outfile)


def export_feature_scores(infile: str, outfile: str = None):
    """
    Export feature score plots from a PyProphet input file.
    
    This function creates plots showing the distribution of feature scores
    (var_* columns) at different levels (ms1, ms2, transition, alignment)
    colored by target/decoy status. Works with OSW, Parquet, and Split Parquet files.
    
    Parameters
    ----------
    infile : str
        Path to the input file (OSW, Parquet, or Split Parquet format).
    outfile : str, optional
        Path to the output PDF file. If None, will be auto-generated based on input filename.
    """
    # Detect file type based on extension and existence
    if infile.endswith(".osw"):
        file_type = "osw"
    elif infile.endswith(".parquet"):
        file_type = "parquet"
    elif os.path.isdir(infile):
        # Check if it's a split parquet directory
        precursor_file = os.path.join(infile, "precursors_features.parquet")
        if os.path.exists(precursor_file):
            file_type = "parquet_split"
        else:
            raise ValueError(f"Directory {infile} does not appear to be a valid split parquet directory")
    else:
        raise ValueError(f"Unsupported file type for {infile}")
    
    logger.info(f"Detected file type: {file_type}")
    
    # Generate output filename if not provided
    if outfile is None:
        if file_type == "osw":
            outfile = infile.replace(".osw", "_feature_scores.pdf")
        elif file_type == "parquet":
            outfile = infile.replace(".parquet", "_feature_scores.pdf")
        else:  # parquet_split
            outfile = infile.rstrip("/") + "_feature_scores.pdf"
    
    logger.info(f"Output file: {outfile}")
    
    # Create config and reader based on file type
    config = ExportIOConfig(
        infile=infile,
        outfile=outfile,
        subsample_ratio=1.0,
        level="export",
        context="export_feature_scores",
    )
    
    # Get appropriate reader
    reader = ReaderDispatcher.get_reader(config)
    
    # Export feature scores using the reader's method
    reader.export_feature_scores(outfile, _plot_feature_scores)
    
    logger.info(f"Feature score plots exported to {outfile}")



def _plot_feature_scores(df: pd.DataFrame, outfile: str, level: str, append: bool = False, sample_size: int = 100000):
    """
    Create plots for feature scores at a specific level.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature scores and DECOY column.
    outfile : str
        Path to the output PDF file.
    level : str
        Level name (ms1, ms2, transition, or alignment).
    append : bool
        If True, append to existing PDF. If False, create new PDF.
    sample_size : int
        Maximum number of rows to use for plotting. If df has more rows,
        a stratified sample (by DECOY) will be taken to reduce memory usage.
    """
    # Get all columns that contain feature scores (VAR_ columns or columns with _VAR_ in name)
    score_cols = [col for col in df.columns if "VAR_" in col.upper() and col != "DECOY"]
    
    if not score_cols:
        logger.warning(f"No feature score columns found for {level} level")
        return
    
    logger.info(f"Found {len(score_cols)} feature score columns for {level} level: {score_cols}")
    
    # Prepare data for plotting - ensure DECOY column exists
    if "DECOY" not in df.columns:
        logger.warning(f"No DECOY column found for {level} level, skipping")
        return
    
    # Only select the columns we need for plotting
    plot_df = df[score_cols + ["DECOY"]].dropna(subset=["DECOY"]).copy()
    
    # Check if we have any data left after dropping NAs
    if len(plot_df) == 0:
        logger.warning(f"No valid data rows found for {level} level after removing rows with null DECOY values")
        return
    
    # Memory optimization: Sample data if it's too large
    if len(plot_df) > sample_size:
        logger.info(f"Dataset has {len(plot_df)} rows, sampling {sample_size} rows (stratified by DECOY) to reduce memory usage")
        # Stratified sampling to maintain target/decoy ratio
        target_df = plot_df[plot_df["DECOY"] == 0]
        decoy_df = plot_df[plot_df["DECOY"] == 1]
        
        # Calculate sample sizes proportional to original distribution
        n_targets = len(target_df)
        n_decoys = len(decoy_df)
        total = n_targets + n_decoys
        
        # Handle edge cases where one group might be empty
        if total == 0:
            logger.warning(f"No data with valid DECOY values for {level} level")
            return
        
        target_sample_size = int(sample_size * n_targets / total) if n_targets > 0 else 0
        decoy_sample_size = int(sample_size * n_decoys / total) if n_decoys > 0 else 0
        
        # Sample from each group
        samples = []
        if n_targets > 0:
            if n_targets > target_sample_size and target_sample_size > 0:
                target_sample = target_df.sample(n=target_sample_size, random_state=42)
            else:
                target_sample = target_df
            samples.append(target_sample)
            
        if n_decoys > 0:
            if n_decoys > decoy_sample_size and decoy_sample_size > 0:
                decoy_sample = decoy_df.sample(n=decoy_sample_size, random_state=42)
            else:
                decoy_sample = decoy_df
            samples.append(decoy_sample)
        
        # Combine samples
        plot_df = pd.concat(samples, ignore_index=True)
        logger.info(f"Sampled {len(plot_df)} rows ({len(samples[0]) if len(samples) > 0 and n_targets > 0 else 0} targets, {len(samples[-1]) if len(samples) > 0 and n_decoys > 0 else 0} decoys)")
    
    # Ensure DECOY is 0 or 1
    if plot_df["DECOY"].dtype == bool:
        plot_df["DECOY"] = plot_df["DECOY"].astype(int)
    
    # Generate a temporary output file for this level
    temp_outfile = outfile.replace(".pdf", f"_{level}_temp.pdf")
    
    # Rename columns to match plot_scores expectations
    # plot_scores expects columns named "SCORE", "MAIN_VAR_*", or "VAR_*"
    rename_dict = {}
    for i, col in enumerate(score_cols):
        # Ensure column names start with VAR_
        if not col.upper().startswith("VAR_"):
            # Extract the var part from column names like FEATURE_MS1_VAR_XXX
            parts = col.split("VAR_")
            if len(parts) > 1:
                new_name = "VAR_" + parts[-1]
            else:
                new_name = "VAR_" + col
            rename_dict[col] = new_name
    
    if rename_dict:
        plot_df.rename(columns=rename_dict, inplace=True)
    
    # Call plot_scores with the formatted dataframe
    plot_scores(plot_df, temp_outfile)
    
    # If appending, merge PDFs, otherwise just rename
    if append and os.path.exists(outfile):
        from pypdf import PdfReader, PdfWriter
        
        # Merge the PDFs
        writer = PdfWriter()
        
        # Add pages from existing PDF
        with open(outfile, "rb") as f:
            existing_pdf = PdfReader(f)
            for page in existing_pdf.pages:
                writer.add_page(page)
        
        # Add pages from new PDF
        with open(temp_outfile, "rb") as f:
            new_pdf = PdfReader(f)
            for page in new_pdf.pages:
                writer.add_page(page)
        
        # Write merged PDF
        with open(outfile, "wb") as f:
            writer.write(f)
        
        # Remove temporary file
        os.remove(temp_outfile)
    else:
        # Just rename temporary file to output file
        if os.path.exists(outfile):
            os.remove(outfile)
        os.rename(temp_outfile, outfile)
