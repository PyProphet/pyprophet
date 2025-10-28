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
    
    # Export feature scores based on file type
    if file_type == "osw":
        _export_feature_scores_osw(infile, outfile)
    elif file_type == "parquet":
        _export_feature_scores_parquet(infile, outfile)
    else:  # parquet_split
        _export_feature_scores_split_parquet(infile, outfile)
    
    logger.info(f"Feature score plots exported to {outfile}")


def _export_feature_scores_osw(infile: str, outfile: str):
    """
    Export feature scores from an OSW file.
    
    Parameters
    ----------
    infile : str
        Path to the OSW input file.
    outfile : str
        Path to the output PDF file.
    """
    con = sqlite3.connect(infile)
    
    try:
        # Process MS1 level if available
        if check_sqlite_table(con, "FEATURE_MS1"):
            logger.info("Processing MS1 level feature scores")
            # Get only VAR_ columns to reduce memory usage
            cursor = con.cursor()
            cursor.execute("PRAGMA table_info(FEATURE_MS1)")
            all_cols = [row[1] for row in cursor.fetchall()]
            var_cols = [col for col in all_cols if "VAR_" in col.upper()]
            
            if var_cols:
                var_cols_sql = ", ".join([f"FEATURE_MS1.{col}" for col in var_cols])
                ms1_query = f"""
                    SELECT 
                        {var_cols_sql},
                        PRECURSOR.DECOY
                    FROM FEATURE_MS1
                    INNER JOIN FEATURE ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
                    INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                """
                df_ms1 = pd.read_sql_query(ms1_query, con)
                if not df_ms1.empty:
                    _plot_feature_scores(df_ms1, outfile, "ms1", append=False)
            else:
                logger.warning("No VAR_ columns found in FEATURE_MS1 table")
        
        # Process MS2 level if available
        if check_sqlite_table(con, "FEATURE_MS2"):
            logger.info("Processing MS2 level feature scores")
            # Get only VAR_ columns to reduce memory usage
            cursor = con.cursor()
            cursor.execute("PRAGMA table_info(FEATURE_MS2)")
            all_cols = [row[1] for row in cursor.fetchall()]
            var_cols = [col for col in all_cols if "VAR_" in col.upper()]
            
            if var_cols:
                var_cols_sql = ", ".join([f"FEATURE_MS2.{col}" for col in var_cols])
                ms2_query = f"""
                    SELECT 
                        {var_cols_sql},
                        PRECURSOR.DECOY
                    FROM FEATURE_MS2
                    INNER JOIN FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
                    INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                """
                df_ms2 = pd.read_sql_query(ms2_query, con)
                if not df_ms2.empty:
                    append = check_sqlite_table(con, "FEATURE_MS1")
                    _plot_feature_scores(df_ms2, outfile, "ms2", append=append)
            else:
                logger.warning("No VAR_ columns found in FEATURE_MS2 table")
        
        # Process transition level if available
        if check_sqlite_table(con, "FEATURE_TRANSITION"):
            logger.info("Processing transition level feature scores")
            # Get only VAR_ columns to reduce memory usage
            cursor = con.cursor()
            cursor.execute("PRAGMA table_info(FEATURE_TRANSITION)")
            all_cols = [row[1] for row in cursor.fetchall()]
            var_cols = [col for col in all_cols if "VAR_" in col.upper()]
            
            if var_cols:
                var_cols_sql = ", ".join([f"FEATURE_TRANSITION.{col}" for col in var_cols])
                transition_query = f"""
                    SELECT 
                        {var_cols_sql},
                        TRANSITION.DECOY
                    FROM FEATURE_TRANSITION
                    INNER JOIN FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
                    INNER JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                """
                df_transition = pd.read_sql_query(transition_query, con)
                if not df_transition.empty:
                    append = check_sqlite_table(con, "FEATURE_MS1") or check_sqlite_table(con, "FEATURE_MS2")
                    _plot_feature_scores(df_transition, outfile, "transition", append=append)
            else:
                logger.warning("No VAR_ columns found in FEATURE_TRANSITION table")
        
        # Process alignment level if available
        if check_sqlite_table(con, "FEATURE_MS2_ALIGNMENT"):
            logger.info("Processing alignment level feature scores")
            # Get only VAR_ columns to reduce memory usage
            cursor = con.cursor()
            cursor.execute("PRAGMA table_info(FEATURE_MS2_ALIGNMENT)")
            all_cols = [row[1] for row in cursor.fetchall()]
            var_cols = [col for col in all_cols if "VAR_" in col.upper()]
            
            if var_cols:
                var_cols_sql = ", ".join(var_cols)
                alignment_query = f"""
                    SELECT 
                        {var_cols_sql},
                        LABEL AS DECOY
                    FROM FEATURE_MS2_ALIGNMENT
                """
                df_alignment = pd.read_sql_query(alignment_query, con)
                if not df_alignment.empty:
                    append = (check_sqlite_table(con, "FEATURE_MS1") or 
                             check_sqlite_table(con, "FEATURE_MS2") or 
                             check_sqlite_table(con, "FEATURE_TRANSITION"))
                    _plot_feature_scores(df_alignment, outfile, "alignment", append=append)
            else:
                logger.warning("No VAR_ columns found in FEATURE_MS2_ALIGNMENT table")
    
    finally:
        con.close()


def _export_feature_scores_parquet(infile: str, outfile: str):
    """
    Export feature scores from a Parquet file.
    
    Parameters
    ----------
    infile : str
        Path to the Parquet input file.
    outfile : str
        Path to the output PDF file.
    """
    logger.info(f"Reading parquet file: {infile}")
    # First, read only column names to identify what to load
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(infile)
    all_columns = parquet_file.schema.names
    
    # Identify columns to read for each level
    ms1_cols = [col for col in all_columns if col.startswith("FEATURE_MS1_VAR_")]
    ms2_cols = [col for col in all_columns if col.startswith("FEATURE_MS2_VAR_")]
    transition_cols = [col for col in all_columns if col.startswith("FEATURE_TRANSITION_VAR_")]
    
    # Determine which columns to read (only what we need)
    cols_to_read = set()
    if ms1_cols and "PRECURSOR_DECOY" in all_columns:
        cols_to_read.update(ms1_cols)
        cols_to_read.add("PRECURSOR_DECOY")
    if ms2_cols and "PRECURSOR_DECOY" in all_columns:
        cols_to_read.update(ms2_cols)
        cols_to_read.add("PRECURSOR_DECOY")
    if transition_cols and "TRANSITION_DECOY" in all_columns:
        cols_to_read.update(transition_cols)
        cols_to_read.add("TRANSITION_DECOY")
    
    if not cols_to_read:
        logger.warning("No VAR_ columns found in parquet file")
        return
    
    # Read only the columns we need
    logger.info(f"Reading {len(cols_to_read)} columns from parquet file")
    df = pd.read_parquet(infile, columns=list(cols_to_read))
    
    # Process MS1 level
    if ms1_cols and "PRECURSOR_DECOY" in df.columns:
        logger.info("Processing MS1 level feature scores")
        ms1_df = df[ms1_cols + ["PRECURSOR_DECOY"]].copy()
        ms1_df.rename(columns={"PRECURSOR_DECOY": "DECOY"}, inplace=True)
        _plot_feature_scores(ms1_df, outfile, "ms1", append=False)
        del ms1_df  # Free memory
    
    # Process MS2 level
    if ms2_cols and "PRECURSOR_DECOY" in df.columns:
        logger.info("Processing MS2 level feature scores")
        ms2_df = df[ms2_cols + ["PRECURSOR_DECOY"]].copy()
        ms2_df.rename(columns={"PRECURSOR_DECOY": "DECOY"}, inplace=True)
        append = bool(ms1_cols)
        _plot_feature_scores(ms2_df, outfile, "ms2", append=append)
        del ms2_df  # Free memory
    
    # Process transition level
    if transition_cols and "TRANSITION_DECOY" in df.columns:
        logger.info("Processing transition level feature scores")
        transition_df = df[transition_cols + ["TRANSITION_DECOY"]].copy()
        transition_df.rename(columns={"TRANSITION_DECOY": "DECOY"}, inplace=True)
        append = bool(ms1_cols or ms2_cols)
        _plot_feature_scores(transition_df, outfile, "transition", append=append)
        del transition_df  # Free memory


def _export_feature_scores_split_parquet(infile: str, outfile: str):
    """
    Export feature scores from a split Parquet directory.
    
    Parameters
    ----------
    infile : str
        Path to the split Parquet directory.
    outfile : str
        Path to the output PDF file.
    """
    # Read precursor features - only necessary columns
    precursor_file = os.path.join(infile, "precursors_features.parquet")
    logger.info(f"Reading precursor features from: {precursor_file}")
    
    # First check what columns are available
    import pyarrow.parquet as pq
    precursor_parquet = pq.ParquetFile(precursor_file)
    all_columns = precursor_parquet.schema.names
    
    # Identify columns to read
    ms1_cols = [col for col in all_columns if col.startswith("FEATURE_MS1_VAR_")]
    ms2_cols = [col for col in all_columns if col.startswith("FEATURE_MS2_VAR_")]
    
    cols_to_read = set()
    if ms1_cols and "PRECURSOR_DECOY" in all_columns:
        cols_to_read.update(ms1_cols)
        cols_to_read.add("PRECURSOR_DECOY")
    if ms2_cols and "PRECURSOR_DECOY" in all_columns:
        cols_to_read.update(ms2_cols)
        cols_to_read.add("PRECURSOR_DECOY")
    
    if cols_to_read:
        logger.info(f"Reading {len(cols_to_read)} columns from precursor features")
        df_precursor = pd.read_parquet(precursor_file, columns=list(cols_to_read))
        
        # Process MS1 level
        if ms1_cols and "PRECURSOR_DECOY" in df_precursor.columns:
            logger.info("Processing MS1 level feature scores")
            ms1_df = df_precursor[ms1_cols + ["PRECURSOR_DECOY"]].copy()
            ms1_df.rename(columns={"PRECURSOR_DECOY": "DECOY"}, inplace=True)
            _plot_feature_scores(ms1_df, outfile, "ms1", append=False)
            del ms1_df  # Free memory
        
        # Process MS2 level
        if ms2_cols and "PRECURSOR_DECOY" in df_precursor.columns:
            logger.info("Processing MS2 level feature scores")
            ms2_df = df_precursor[ms2_cols + ["PRECURSOR_DECOY"]].copy()
            ms2_df.rename(columns={"PRECURSOR_DECOY": "DECOY"}, inplace=True)
            append = bool(ms1_cols)
            _plot_feature_scores(ms2_df, outfile, "ms2", append=append)
            del ms2_df  # Free memory
        
        del df_precursor  # Free memory
    
    # Read transition features if available
    transition_file = os.path.join(infile, "transition_features.parquet")
    if os.path.exists(transition_file):
        logger.info(f"Reading transition features from: {transition_file}")
        
        # Check what columns are available
        transition_parquet = pq.ParquetFile(transition_file)
        transition_all_columns = transition_parquet.schema.names
        transition_cols = [col for col in transition_all_columns if col.startswith("FEATURE_TRANSITION_VAR_")]
        
        if transition_cols and "TRANSITION_DECOY" in transition_all_columns:
            # Read only necessary columns
            cols_to_read = transition_cols + ["TRANSITION_DECOY"]
            logger.info(f"Reading {len(cols_to_read)} columns from transition features")
            df_transition = pd.read_parquet(transition_file, columns=cols_to_read)
            
            logger.info("Processing transition level feature scores")
            transition_df = df_transition[transition_cols + ["TRANSITION_DECOY"]].copy()
            transition_df.rename(columns={"TRANSITION_DECOY": "DECOY"}, inplace=True)
            append = bool(ms1_cols or ms2_cols)
            _plot_feature_scores(transition_df, outfile, "transition", append=append)
            del transition_df, df_transition  # Free memory
    
    # Read alignment features if available
    alignment_file = os.path.join(infile, "feature_alignment.parquet")
    if os.path.exists(alignment_file):
        logger.info(f"Reading alignment features from: {alignment_file}")
        
        # Check what columns are available
        alignment_parquet = pq.ParquetFile(alignment_file)
        alignment_all_columns = alignment_parquet.schema.names
        var_cols = [col for col in alignment_all_columns if col.startswith("VAR_")]
        
        if var_cols and "DECOY" in alignment_all_columns:
            # Read only necessary columns
            cols_to_read = var_cols + ["DECOY"]
            logger.info(f"Reading {len(cols_to_read)} columns from alignment features")
            df_alignment = pd.read_parquet(alignment_file, columns=cols_to_read)
            
            logger.info("Processing alignment level feature scores")
            alignment_df = df_alignment[var_cols + ["DECOY"]].copy()
            append = bool(ms1_cols or ms2_cols or (os.path.exists(transition_file) and transition_cols))
            _plot_feature_scores(alignment_df, outfile, "alignment", append=append)
            del alignment_df, df_alignment  # Free memory


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
        
        target_sample_size = int(sample_size * n_targets / total)
        decoy_sample_size = int(sample_size * n_decoys / total)
        
        # Sample from each group
        if n_targets > target_sample_size:
            target_sample = target_df.sample(n=target_sample_size, random_state=42)
        else:
            target_sample = target_df
            
        if n_decoys > decoy_sample_size:
            decoy_sample = decoy_df.sample(n=decoy_sample_size, random_state=42)
        else:
            decoy_sample = decoy_df
        
        # Combine samples
        plot_df = pd.concat([target_sample, decoy_sample], ignore_index=True)
        logger.info(f"Sampled {len(plot_df)} rows ({len(target_sample)} targets, {len(decoy_sample)} decoys)")
    
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
    
    # plot_scores requires a "SCORE" column - use the first VAR_ column as SCORE
    var_cols = [col for col in plot_df.columns if col.startswith("VAR_")]
    if var_cols and "SCORE" not in plot_df.columns:
        # Add SCORE column as a copy of the first VAR_ column
        plot_df["SCORE"] = plot_df[var_cols[0]]
    
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
