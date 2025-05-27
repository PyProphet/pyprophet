import pandas as pd
from .report import post_scoring_report
from .io.util import get_parquet_column_names


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
