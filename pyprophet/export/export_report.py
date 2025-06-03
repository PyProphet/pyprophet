import sqlite3
import pandas as pd

from ..report import post_scoring_report
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

    cols_infile = get_parquet_column_names(infile)

    select_cols = [
        "RUN_ID",
        "PROTEIN_ID",
        "PEPTIDE_ID",
        "PRECURSOR_ID",
        "PRECURSOR_DECOY",
        "FEATURE_MS2_AREA_INTENSITY",
        "SCORE_MS2_SCORE",
        "SCORE_MS2_PEAK_GROUP_RANK",
        "SCORE_MS2_Q_VALUE",
        "SCORE_PEPTIDE_GLOBAL_SCORE",
        "SCORE_PEPTIDE_GLOBAL_Q_VALUE",
        "SCORE_PEPTIDE_EXPERIMENT_WIDE_SCORE",
        "SCORE_PEPTIDE_EXPERIMENT_WIDE_Q_VALUE",
        "SCORE_PEPTIDE_RUN_SPECIFIC_SCORE",
        "SCORE_PEPTIDE_RUN_SPECIFIC_Q_VALUE",
        "SCORE_PROTEIN_GLOBAL_SCORE",
        "SCORE_PROTEIN_GLOBAL_Q_VALUE",
        "SCORE_PROTEIN_EXPERIMENT_WIDE_SCORE",
        "SCORE_PROTEIN_EXPERIMENT_WIDE_Q_VALUE",
        "SCORE_IPF_QVALUE",
    ]

    # Filter select cols based on available columns in the input file
    select_cols = [col for col in select_cols if col in cols_infile]

    # Load the input data
    df = pd.read_parquet(infile, columns=select_cols)

    post_scoring_report(df, outfile)
