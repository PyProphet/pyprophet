# encoding: latin-1

import pandas as pd
pd.options.display.width = 220
pd.options.display.precision = 6

import numpy as np
import sqlite3
from .data_handling import filterChromByLabels, check_sqlite_table
from .std_logger import logging
from .report import plot_scores

def export_tsv(infile, outfile, format, outcsv, transition_quantification, ipf, max_rs_peakgroup_qvalue, peptide, max_global_peptide_qvalue, protein, max_global_protein_qvalue):

    con = sqlite3.connect(infile)

    ipf_present = False
    if ipf:
        ipf_present = check_sqlite_table(con, "SCORE_IPF")

    if ipf_present and ipf:
        if transition_quantification:
            data = pd.read_sql_query("SELECT RUN.ID AS id_run, PEPTIDE.ID AS id_peptide, PEPTIDE_IPF.MODIFIED_SEQUENCE || '_' || PRECURSOR.ID AS transition_group_id, PRECURSOR.DECOY AS decoy, RUN.ID AS run_id, RUN.FILENAME AS filename, FEATURE.EXP_RT AS RT, FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt, FEATURE.DELTA_RT AS delta_rt, FEATURE.NORM_RT AS iRT, PRECURSOR.LIBRARY_RT AS assay_iRT, FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT, FEATURE.ID AS id, PEPTIDE_IPF.UNMODIFIED_SEQUENCE AS Sequence, PEPTIDE_IPF.MODIFIED_SEQUENCE AS FullUniModPeptideName, PRECURSOR.CHARGE AS Charge, PRECURSOR.PRECURSOR_MZ AS mz, FEATURE_MS2.AREA_INTENSITY AS Intensity, FEATURE_MS1.AREA_INTENSITY AS aggr_prec_peak_area, FEATURE_MS1.APEX_INTENSITY AS aggr_prec_peak_apex, aggr_peak_area, aggr_peak_apex, aggr_fragment_annotation, FEATURE.LEFT_WIDTH AS leftWidth, FEATURE.RIGHT_WIDTH AS rightWidth, SCORE_MS1.PEP AS ms1_pep, SCORE_MS2.PEP AS ms2_pep, SCORE_IPF.PRECURSOR_PEAKGROUP_PEP AS precursor_pep, SCORE_IPF.PEP AS ipf_pep, SCORE_MS2.RANK AS peak_group_rank, SCORE_MS2.SCORE AS d_score, SCORE_MS2.QVALUE AS ms2_m_score, SCORE_IPF.QVALUE AS m_score FROM PRECURSOR JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID JOIN RUN ON RUN.ID = FEATURE.RUN_ID JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID JOIN (SELECT FEATURE_ID, GROUP_CONCAT(AREA_INTENSITY,';') AS aggr_peak_area, GROUP_CONCAT(APEX_INTENSITY,';') AS aggr_peak_apex, GROUP_CONCAT(TRANSITION_ID,';') AS aggr_fragment_annotation FROM FEATURE_TRANSITION INNER JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID WHERE TRANSITION.DECOY == 0 GROUP BY FEATURE_ID) AS FEATURE_TRANSITION ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID JOIN SCORE_MS1 ON SCORE_MS1.FEATURE_ID = FEATURE.ID JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID JOIN SCORE_IPF ON SCORE_IPF.FEATURE_ID = FEATURE.ID JOIN PEPTIDE AS PEPTIDE_IPF ON SCORE_IPF.PEPTIDE_ID = PEPTIDE_IPF.ID ORDER BY transition_group_id, peak_group_rank;", con)
        else:
            data = pd.read_sql_query("SELECT RUN.ID AS id_run, PEPTIDE.ID AS id_peptide, PEPTIDE_IPF.MODIFIED_SEQUENCE || '_' || PRECURSOR.ID AS transition_group_id, PRECURSOR.DECOY AS decoy, RUN.ID AS run_id, RUN.FILENAME AS filename, FEATURE.EXP_RT AS RT, FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt, FEATURE.DELTA_RT AS delta_rt, FEATURE.NORM_RT AS iRT, PRECURSOR.LIBRARY_RT AS assay_iRT, FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT, FEATURE.ID AS id, PEPTIDE_IPF.UNMODIFIED_SEQUENCE AS Sequence, PEPTIDE_IPF.MODIFIED_SEQUENCE AS FullUniModPeptideName, PRECURSOR.CHARGE AS Charge, PRECURSOR.PRECURSOR_MZ AS mz, FEATURE_MS2.AREA_INTENSITY AS Intensity, FEATURE_MS1.AREA_INTENSITY AS aggr_prec_peak_area, FEATURE_MS1.APEX_INTENSITY AS aggr_prec_peak_apex, FEATURE.LEFT_WIDTH AS leftWidth, FEATURE.RIGHT_WIDTH AS rightWidth, SCORE_MS1.PEP AS ms1_pep, SCORE_MS2.PEP AS ms2_pep, SCORE_IPF.PRECURSOR_PEAKGROUP_PEP AS precursor_pep, SCORE_IPF.PEP AS ipf_pep, SCORE_MS2.RANK AS peak_group_rank, SCORE_MS2.SCORE AS d_score, SCORE_MS2.QVALUE AS ms2_m_score, SCORE_IPF.QVALUE AS m_score FROM PRECURSOR JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID JOIN RUN ON RUN.ID = FEATURE.RUN_ID JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID JOIN SCORE_MS1 ON SCORE_MS1.FEATURE_ID = FEATURE.ID JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID JOIN SCORE_IPF ON SCORE_IPF.FEATURE_ID = FEATURE.ID JOIN PEPTIDE AS PEPTIDE_IPF ON SCORE_IPF.PEPTIDE_ID = PEPTIDE_IPF.ID ORDER BY transition_group_id, peak_group_rank;", con)

    else:
        if transition_quantification:
            data = pd.read_sql_query("SELECT RUN.ID AS id_run, PEPTIDE.ID AS id_peptide, PRECURSOR.ID AS transition_group_id, PRECURSOR.DECOY AS decoy, RUN.ID AS run_id, RUN.FILENAME AS filename, FEATURE.EXP_RT AS RT, FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt, FEATURE.DELTA_RT AS delta_rt, FEATURE.NORM_RT AS iRT, PRECURSOR.LIBRARY_RT AS assay_iRT, FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT, FEATURE.ID AS id, PEPTIDE.UNMODIFIED_SEQUENCE AS Sequence, PEPTIDE.MODIFIED_SEQUENCE AS FullUniModPeptideName, PRECURSOR.CHARGE AS Charge, PRECURSOR.PRECURSOR_MZ AS mz, FEATURE_MS2.AREA_INTENSITY AS Intensity, FEATURE_MS1.AREA_INTENSITY AS aggr_prec_peak_area, FEATURE_MS1.APEX_INTENSITY AS aggr_prec_peak_apex, aggr_peak_area, aggr_peak_apex, aggr_fragment_annotation, FEATURE.LEFT_WIDTH AS leftWidth, FEATURE.RIGHT_WIDTH AS rightWidth, SCORE_MS2.RANK AS peak_group_rank, SCORE_MS2.SCORE AS d_score, SCORE_MS2.QVALUE AS m_score FROM PRECURSOR JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID JOIN RUN ON RUN.ID = FEATURE.RUN_ID JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID JOIN (SELECT FEATURE_ID, GROUP_CONCAT(AREA_INTENSITY,';') AS aggr_peak_area, GROUP_CONCAT(APEX_INTENSITY,';') AS aggr_peak_apex, GROUP_CONCAT(TRANSITION_ID,';') AS aggr_fragment_annotation FROM FEATURE_TRANSITION INNER JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID GROUP BY FEATURE_ID) AS FEATURE_TRANSITION ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID ORDER BY transition_group_id, peak_group_rank;", con)
        else:
            data = pd.read_sql_query("SELECT RUN.ID AS id_run, PEPTIDE.ID AS id_peptide, PRECURSOR.ID AS transition_group_id, PRECURSOR.DECOY AS decoy, RUN.ID AS run_id, RUN.FILENAME AS filename, FEATURE.EXP_RT AS RT, FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt, FEATURE.DELTA_RT AS delta_rt, FEATURE.NORM_RT AS iRT, PRECURSOR.LIBRARY_RT AS assay_iRT, FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT, FEATURE.ID AS id, PEPTIDE.UNMODIFIED_SEQUENCE AS Sequence, PEPTIDE.MODIFIED_SEQUENCE AS FullUniModPeptideName, PRECURSOR.CHARGE AS Charge, PRECURSOR.PRECURSOR_MZ AS mz, FEATURE_MS2.AREA_INTENSITY AS Intensity, FEATURE_MS1.AREA_INTENSITY AS aggr_prec_peak_area, FEATURE_MS1.APEX_INTENSITY AS aggr_prec_peak_apex, FEATURE.LEFT_WIDTH AS leftWidth, FEATURE.RIGHT_WIDTH AS rightWidth, SCORE_MS2.RANK AS peak_group_rank, SCORE_MS2.SCORE AS d_score, SCORE_MS2.QVALUE AS m_score FROM PRECURSOR JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID JOIN RUN ON RUN.ID = FEATURE.RUN_ID JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID ORDER BY transition_group_id, peak_group_rank;", con)

    # Append concatenated protein identifier
    data_protein = pd.read_sql_query("SELECT PEPTIDE_ID AS id_peptide, GROUP_CONCAT(PROTEIN.PROTEIN_ACCESSION,';') AS ProteinName FROM PEPTIDE_PROTEIN_MAPPING INNER JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID GROUP BY PEPTIDE_ID;", con)
    data = pd.merge(data, data_protein, how='inner', on=['id_peptide'])

    # Append peptide error-rate control
    peptide_present = False
    if peptide:
        peptide_present = check_sqlite_table(con, "SCORE_PEPTIDE")

    if peptide_present and peptide:
        data_peptide_run = pd.read_sql_query("SELECT RUN_ID AS id_run, PEPTIDE_ID AS id_peptide, QVALUE AS m_score_peptide_run_specific FROM SCORE_PEPTIDE WHERE CONTEXT == 'run-specific';", con)
        if len(data_peptide_run.index) > 0:
            data = pd.merge(data, data_peptide_run, how='inner', on=['id_run','id_peptide'])

        data_peptide_experiment = pd.read_sql_query("SELECT RUN_ID AS id_run, PEPTIDE_ID AS id_peptide, QVALUE AS m_score_peptide_experiment_wide FROM SCORE_PEPTIDE WHERE CONTEXT == 'experiment-wide';", con)
        if len(data_peptide_experiment.index) > 0:
            data = pd.merge(data, data_peptide_experiment, on=['id_run','id_peptide'])

        data_peptide_global = pd.read_sql_query("SELECT PEPTIDE_ID AS id_peptide, QVALUE AS m_score_peptide_global FROM SCORE_PEPTIDE WHERE CONTEXT == 'global';", con)
        if len(data_peptide_global.index) > 0:
            data = pd.merge(data, data_peptide_global[data_peptide_global['m_score_peptide_global'] < max_global_peptide_qvalue], on=['id_peptide'])

    # Append protein error-rate control
    protein_present = False
    if protein:
        protein_present = check_sqlite_table(con, "SCORE_PROTEIN")

    if protein_present and protein:
        data_protein_run = pd.read_sql_query("SELECT RUN_ID AS id_run, PEPTIDE_ID AS id_peptide, MIN(QVALUE) AS m_score_protein_run_specific FROM PEPTIDE_PROTEIN_MAPPING INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID WHERE CONTEXT == 'run-specific' GROUP BY RUN_ID, PEPTIDE_ID;", con)
        if len(data_protein_run.index) > 0:
            data = pd.merge(data, data_protein_run, how='inner', on=['id_run','id_peptide'])

        data_protein_experiment = pd.read_sql_query("SELECT RUN_ID AS id_run, PEPTIDE_ID AS id_peptide, MIN(QVALUE) AS m_score_protein_experiment_wide FROM PEPTIDE_PROTEIN_MAPPING INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID WHERE CONTEXT == 'experiment-wide' GROUP BY RUN_ID, PEPTIDE_ID;", con)
        if len(data_protein_experiment.index) > 0:
            data = pd.merge(data, data_protein_experiment, how='inner', on=['id_run','id_peptide'])

        data_protein_global = pd.read_sql_query("SELECT PEPTIDE_ID AS id_peptide, MIN(QVALUE) AS m_score_protein_global FROM PEPTIDE_PROTEIN_MAPPING INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID WHERE CONTEXT == 'global' GROUP BY PEPTIDE_ID;", con)
        if len(data_protein_global.index) > 0:
            data = pd.merge(data, data_protein_global[data_protein_global['m_score_protein_global'] < max_global_protein_qvalue], how='inner', on=['id_peptide'])

    if outcsv:
        sep = ","
    else:
        sep = "\t"

    if format == 'legacy':
        data.drop(['id_run','id_peptide'], axis=1).to_csv(outfile, sep=sep, index=False)
    elif format == 'matrix':
        # select top ranking peak group only
        data = data.iloc[data.groupby(['run_id','transition_group_id']).apply(lambda x: x['m_score'].idxmin())]
        # limit peak groups to q-value cutoff
        data = data[data['m_score'] < max_rs_peakgroup_qvalue]
        # restructure dataframe to matrix
        data = data[['transition_group_id','Sequence','FullUniModPeptideName','ProteinName','filename','Intensity']]
        data = data.pivot_table(index=['transition_group_id','Sequence','FullUniModPeptideName','ProteinName'], columns='filename', values='Intensity')
        data.to_csv(outfile, sep=sep, index=True)

    con.close()

def export_score_plots(infile):

    con = sqlite3.connect(infile)

    if check_sqlite_table(con, "SCORE_MS2"):
        outfile = infile.split(".osw")[0] + "_ms2_score_plots.pdf"
        table_ms2 = pd.read_sql_query("SELECT *, RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_MS2 INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID, EXP_RT FROM FEATURE) AS FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.ID INNER JOIN (SELECT ID, DECOY FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID WHERE RANK == 1 ORDER BY RUN_ID, PRECURSOR.ID ASC, FEATURE.EXP_RT ASC;", con)
        plot_scores(table_ms2, outfile)

    if check_sqlite_table(con, "SCORE_MS1"):
        outfile = infile.split(".osw")[0] + "_ms1_score_plots.pdf"
        table_ms1 = pd.read_sql_query("SELECT *, RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_MS1 INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID, EXP_RT FROM FEATURE) AS FEATURE ON FEATURE_MS1.FEATURE_ID = FEATURE.ID INNER JOIN (SELECT ID, DECOY FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID WHERE RANK == 1 ORDER BY RUN_ID, PRECURSOR.ID ASC, FEATURE.EXP_RT ASC;", con)
        plot_scores(table_ms1, outfile)

    if check_sqlite_table(con, "SCORE_TRANSITION"):
        outfile = infile.split(".osw")[0] + "_transition_score_plots.pdf"
        table_transition = pd.read_sql_query("SELECT TRANSITION.DECOY AS DECOY, FEATURE_TRANSITION.*, SCORE_TRANSITION.*, RUN_ID || '_' || FEATURE_TRANSITION.FEATURE_ID || '_' || PRECURSOR_ID || '_' || FEATURE_TRANSITION.TRANSITION_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_TRANSITION INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID, EXP_RT FROM FEATURE) AS FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID INNER JOIN SCORE_TRANSITION ON FEATURE_TRANSITION.FEATURE_ID = SCORE_TRANSITION.FEATURE_ID AND FEATURE_TRANSITION.TRANSITION_ID = SCORE_TRANSITION.TRANSITION_ID INNER JOIN (SELECT ID, DECOY FROM TRANSITION) AS TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID ORDER BY RUN_ID, PRECURSOR.ID, FEATURE.EXP_RT, TRANSITION.ID;", con)
        plot_scores(table_transition, outfile)

    con.close()

def filter_sqmass(sqmassfiles, infile, max_precursor_pep, max_peakgroup_pep, max_transition_pep):
    con = sqlite3.connect(infile)

    # process each sqmassfile independently
    for sqm_in in sqmassfiles:
        sqm_out = sqm_in.split(".sqMass")[0] + "_filtered.sqMass"

        transitions = pd.read_sql_query("SELECT TRANSITION_ID AS transition_id FROM PRECURSOR INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID INNER JOIN SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID WHERE SCORE_MS1.PEP <=" + str(max_precursor_pep) + " AND SCORE_MS2.PEP <=" + str(max_peakgroup_pep) + " AND SCORE_TRANSITION.PEP <=" + str(max_transition_pep) + " AND FILENAME LIKE '%" + sqm_in.split(".sqMass")[0] + "%';", con)['transition_id'].values

        filterChromByLabels(sqm_in, sqm_out, transitions)

