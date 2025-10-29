import sqlite3
import pandas as pd
from pathlib import Path


from .._config import ExportIOConfig
from ..report import post_scoring_report
from ..io.dispatcher import ReaderDispatcher
from ..io.util import get_parquet_column_names
from ..io.util import check_sqlite_table
from ..report import plot_scores
from loguru import logger


def _check_pyarrow_available():
    """
    Helper function to check if pyarrow is available and provide helpful error message.
    
    Returns
    -------
    module
        The pyarrow.parquet module if available
        
    Raises
    ------
    ImportError
        If pyarrow is not installed
    """
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for parquet file operations. "
            "Install it with: pip install pyarrow or pip install pyprophet[parquet]"
        )


def export_feature_scores(infile, outfile=None):
    """
    Export feature score plots from a PyProphet input file.
    
    This function works with OSW, Parquet, and Split Parquet formats.
    - If SCORE tables exist: applies RANK==1 filtering and plots SCORE + VAR_ columns
    - If SCORE tables don't exist: plots only VAR_ columns
    
    Parameters
    ----------
    infile : str
        Path to input file (OSW, Parquet, or Split Parquet directory)
    outfile : str, optional
        Base path for output PDF files. If None, derives from infile.
    """
    # Determine file type and route to appropriate handler
    inpath = Path(infile)
    
    # Check if it's a directory (split parquet) or file
    if inpath.is_dir():
        # Split parquet directory
        logger.info(f"Detected split parquet directory: {infile}")
        _export_feature_scores_from_split_parquet(infile, outfile)
    elif infile.endswith('.parquet'):
        logger.info(f"Detected parquet file: {infile}")
        _export_feature_scores_from_parquet(infile, outfile)
    elif infile.endswith('.osw'):
        logger.info(f"Detected OSW file: {infile}")
        _export_feature_scores_from_osw(infile, outfile)
    else:
        raise ValueError(f"Unsupported file format: {infile}. Must be .osw, .parquet, or split parquet directory.")


def _export_feature_scores_from_osw(infile, outfile=None):
    """
    Export feature scores from OSW file.
    Detects if SCORE tables exist and adjusts behavior accordingly.
    """
    con = sqlite3.connect(infile)
    
    # Check for SCORE tables
    has_score_ms2 = check_sqlite_table(con, "SCORE_MS2")
    has_score_ms1 = check_sqlite_table(con, "SCORE_MS1")
    has_score_transition = check_sqlite_table(con, "SCORE_TRANSITION")
    
    if has_score_ms2 or has_score_ms1 or has_score_transition:
        logger.info("SCORE tables detected - applying RANK==1 filter and plotting SCORE + VAR_ columns")
    else:
        logger.info("No SCORE tables detected - plotting only VAR_ columns")
    
    # MS2 level
    if check_sqlite_table(con, "FEATURE_MS2"):
        if outfile:
            out_ms2 = outfile.replace('.pdf', '_ms2.pdf') if outfile.endswith('.pdf') else f"{outfile}_ms2.pdf"
        else:
            out_ms2 = infile.split(".osw")[0] + "_ms2_feature_scores.pdf"
        
        if has_score_ms2:
            # Scored mode: Include SCORE columns and apply RANK==1 filter
            query_ms2 = """
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
"""
        else:
            # Unscored mode: Only VAR_ columns, no RANK filter
            query_ms2 = """
SELECT FEATURE_MS2.*,
       FEATURE.RUN_ID,
       FEATURE.PRECURSOR_ID,
       FEATURE.EXP_RT,
       PRECURSOR.CHARGE AS VAR_PRECURSOR_CHARGE,
       PRECURSOR.DECOY,
       VAR_TRANSITION_SCORE.VAR_TRANSITION_NUM_SCORE,
       FEATURE.RUN_ID || '_' || FEATURE.PRECURSOR_ID AS GROUP_ID
FROM FEATURE_MS2
INNER JOIN FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
LEFT JOIN
  (SELECT PRECURSOR_ID AS ID,
          COUNT(*) AS VAR_TRANSITION_NUM_SCORE
   FROM TRANSITION_PRECURSOR_MAPPING
   INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
   WHERE DETECTING==1
   GROUP BY PRECURSOR_ID) AS VAR_TRANSITION_SCORE ON FEATURE.PRECURSOR_ID = VAR_TRANSITION_SCORE.ID
ORDER BY FEATURE.RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
"""
        
        table_ms2 = pd.read_sql_query(query_ms2, con)
        if len(table_ms2) > 0:
            plot_scores(table_ms2, out_ms2)
            logger.info(f"Exported MS2 feature scores to {out_ms2}")
    
    # MS1 level
    if check_sqlite_table(con, "FEATURE_MS1"):
        if outfile:
            out_ms1 = outfile.replace('.pdf', '_ms1.pdf') if outfile.endswith('.pdf') else f"{outfile}_ms1.pdf"
        else:
            out_ms1 = infile.split(".osw")[0] + "_ms1_feature_scores.pdf"
        
        if has_score_ms1:
            # Scored mode
            query_ms1 = """
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
"""
        else:
            # Unscored mode
            query_ms1 = """
SELECT FEATURE_MS1.*,
       FEATURE.RUN_ID,
       FEATURE.PRECURSOR_ID,
       FEATURE.EXP_RT,
       PRECURSOR.CHARGE AS VAR_PRECURSOR_CHARGE,
       PRECURSOR.DECOY,
       FEATURE.RUN_ID || '_' || FEATURE.PRECURSOR_ID AS GROUP_ID
FROM FEATURE_MS1
INNER JOIN FEATURE ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
ORDER BY FEATURE.RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
"""
        
        table_ms1 = pd.read_sql_query(query_ms1, con)
        if len(table_ms1) > 0:
            plot_scores(table_ms1, out_ms1)
            logger.info(f"Exported MS1 feature scores to {out_ms1}")
    
    # Transition level
    if check_sqlite_table(con, "FEATURE_TRANSITION"):
        if outfile:
            out_transition = outfile.replace('.pdf', '_transition.pdf') if outfile.endswith('.pdf') else f"{outfile}_transition.pdf"
        else:
            out_transition = infile.split(".osw")[0] + "_transition_feature_scores.pdf"
        
        if has_score_transition:
            # Scored mode
            query_transition = """
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
"""
        else:
            # Unscored mode
            query_transition = """
SELECT TRANSITION.DECOY AS DECOY,
       FEATURE_TRANSITION.*,
       FEATURE.RUN_ID,
       FEATURE.PRECURSOR_ID,
       FEATURE.EXP_RT,
       PRECURSOR.CHARGE AS VAR_PRECURSOR_CHARGE,
       TRANSITION.CHARGE AS VAR_PRODUCT_CHARGE,
       FEATURE.RUN_ID || '_' || FEATURE_TRANSITION.FEATURE_ID || '_' || FEATURE.PRECURSOR_ID || '_' || FEATURE_TRANSITION.TRANSITION_ID AS GROUP_ID
FROM FEATURE_TRANSITION
INNER JOIN FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
ORDER BY FEATURE.RUN_ID,
         PRECURSOR.ID,
         FEATURE.EXP_RT,
         TRANSITION.ID;
"""
        
        table_transition = pd.read_sql_query(query_transition, con)
        if len(table_transition) > 0:
            plot_scores(table_transition, out_transition)
            logger.info(f"Exported transition feature scores to {out_transition}")
    
    con.close()


def _export_feature_scores_from_parquet(infile, outfile=None):
    """
    Export feature scores from single Parquet file.
    """
    pq = _check_pyarrow_available()
    
    # Read parquet file
    table = pq.read_table(infile)
    df = table.to_pandas()
    
    # Check for SCORE columns
    score_columns = [col for col in df.columns if col.startswith('SCORE_')]
    has_scores = len(score_columns) > 0
    
    if has_scores:
        logger.info("SCORE columns detected - applying RANK==1 filter and plotting SCORE + VAR_ columns")
        # Filter to RANK==1 if SCORE_MS2_PEAK_GROUP_RANK exists
        if 'SCORE_MS2_PEAK_GROUP_RANK' in df.columns:
            df = df[df['SCORE_MS2_PEAK_GROUP_RANK'] == 1].copy()
    else:
        logger.info("No SCORE columns detected - plotting only VAR_ columns")
    
    # Generate GROUP_ID if needed
    if 'GROUP_ID' not in df.columns:
        if 'RUN_ID' in df.columns and 'PRECURSOR_ID' in df.columns:
            df['GROUP_ID'] = df['RUN_ID'].astype(str) + '_' + df['PRECURSOR_ID'].astype(str)
    
    # Determine DECOY column name (could be PRECURSOR_DECOY or DECOY)
    decoy_col = None
    for col in ['DECOY', 'PRECURSOR_DECOY', 'PEPTIDE_DECOY']:
        if col in df.columns:
            decoy_col = col
            break
    
    if decoy_col:
        df = df.rename(columns={decoy_col: 'DECOY'})
    
    # Export plots
    if outfile is None:
        outfile = infile.replace('.parquet', '_feature_scores.pdf')
    
    if len(df) > 0:
        plot_scores(df, outfile)
        logger.info(f"Exported feature scores to {outfile}")


def _export_feature_scores_from_split_parquet(infile, outfile=None):
    """
    Export feature scores from split Parquet directory.
    """
    pq = _check_pyarrow_available()
    
    inpath = Path(infile)
    
    # Read precursor features
    precursor_file = inpath / "precursors_features.parquet"
    if precursor_file.exists():
        table = pq.read_table(str(precursor_file))
        df = table.to_pandas()
        
        # Check for SCORE columns
        score_columns = [col for col in df.columns if col.startswith('SCORE_')]
        has_scores = len(score_columns) > 0
        
        if has_scores:
            logger.info("SCORE columns detected - applying RANK==1 filter and plotting SCORE + VAR_ columns")
            # Filter to RANK==1 if SCORE_MS2_PEAK_GROUP_RANK exists
            if 'SCORE_MS2_PEAK_GROUP_RANK' in df.columns:
                df = df[df['SCORE_MS2_PEAK_GROUP_RANK'] == 1].copy()
        else:
            logger.info("No SCORE columns detected - plotting only VAR_ columns")
        
        # Generate GROUP_ID if needed
        if 'GROUP_ID' not in df.columns:
            if 'RUN_ID' in df.columns and 'PRECURSOR_ID' in df.columns:
                df['GROUP_ID'] = df['RUN_ID'].astype(str) + '_' + df['PRECURSOR_ID'].astype(str)
        
        # Determine DECOY column name
        decoy_col = None
        for col in ['DECOY', 'PRECURSOR_DECOY', 'PEPTIDE_DECOY']:
            if col in df.columns:
                decoy_col = col
                break
        
        if decoy_col:
            df = df.rename(columns={decoy_col: 'DECOY'})
        
        # Export plots
        if outfile is None:
            outfile = str(inpath / "feature_scores.pdf")
        
        if len(df) > 0:
            plot_scores(df, outfile)
            logger.info(f"Exported feature scores to {outfile}")
    else:
        logger.warning(f"Precursor features file not found: {precursor_file}")


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
