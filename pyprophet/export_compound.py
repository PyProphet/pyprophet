import pandas as pd
import sqlite3

from .data_handling import check_sqlite_table
from .report import plot_scores

def export_compound_tsv(infile, outfile, format, outcsv, max_rs_peakgroup_qvalue):
    con = sqlite3.connect(infile)

    if check_sqlite_table(con, "SCORE_MS1"):
        data = pd.read_sql_query("""
                               SELECT
                                   RUN.ID AS id_run,
                                   COMPOUND.ID AS id_compound,
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
                                   COMPOUND.SUM_FORMULA AS sum_formula,
                                   COMPOUND.COMPOUND_NAME AS compound_name,
                                   COMPOUND.ADDUCTS AS Adducts,
                                   PRECURSOR.CHARGE AS Charge,
                                   PRECURSOR.PRECURSOR_MZ AS mz,
                                   FEATURE_MS2.AREA_INTENSITY AS Intensity,
                                   FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
                                   FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
                                   FEATURE.LEFT_WIDTH AS leftWidth,
                                   FEATURE.RIGHT_WIDTH AS rightWidth,
                                   SCORE_MS1.RANK AS peak_group_rank,
                                   SCORE_MS1.SCORE AS d_score,
                                   SCORE_MS1.QVALUE AS m_score
                               FROM PRECURSOR
                               INNER JOIN PRECURSOR_COMPOUND_MAPPING ON PRECURSOR.ID = PRECURSOR_COMPOUND_MAPPING.PRECURSOR_ID
                               INNER JOIN COMPOUND ON PRECURSOR_COMPOUND_MAPPING.COMPOUND_ID = COMPOUND.ID
                               INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                               INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
                               LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
                               LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
                               LEFT JOIN SCORE_MS1 ON SCORE_MS1.FEATURE_ID = FEATURE.ID
                               WHERE SCORE_MS1.QVALUE < %s
                               ORDER BY transition_group_id,
                                        peak_group_rank;
                               """ % max_rs_peakgroup_qvalue, con)
    else:
        data = pd.read_sql_query("""
                               SELECT
                                   RUN.ID AS id_run,
                                   COMPOUND.ID AS id_compound,
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
                                   COMPOUND.SUM_FORMULA AS sum_formula,
                                   COMPOUND.COMPOUND_NAME AS compound_name,
                                   COMPOUND.ADDUCTS AS Adducts,
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
                               INNER JOIN PRECURSOR_COMPOUND_MAPPING ON PRECURSOR.ID = PRECURSOR_COMPOUND_MAPPING.PRECURSOR_ID
                               INNER JOIN COMPOUND ON PRECURSOR_COMPOUND_MAPPING.COMPOUND_ID = COMPOUND.ID
                               INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                               INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
                               LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
                               LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
                               LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
                               WHERE SCORE_MS2.QVALUE < %s
                               ORDER BY transition_group_id,
                                        peak_group_rank;
                               """ % max_rs_peakgroup_qvalue, con)

    con.close()
    
    if outcsv: 
        sep = ","
    else:
        sep = "\t"
    
    # select top ranking peak group
    if format == "legacy_merged":
        data.drop(['id_run','id_compound'], axis=1).to_csv(outfile, sep=sep, index=False)
    elif format == "matrix":
        # select top ranking peak group only
        data = data.iloc[data.groupby(['run_id','transition_group_id']).apply(lambda x: x['m_score'].idxmin())]
        # restructure dataframe to matrix
        data = data[['transition_group_id','sum_formula','compound_name', 'Adducts','filename','Intensity']]
        data = data.pivot_table(index=['transition_group_id','sum_formula','compound_name','Adducts'], columns='filename', values='Intensity')
        data.to_csv(outfile, sep=sep, index=True)

