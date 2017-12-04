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
            data = pd.read_sql_query("select run.id as id_run, peptide.id as id_peptide, protein.id as id_protein, peptide_ipf.modified_sequence || '_' || precursor.id as transition_group_id, precursor.decoy as decoy, run.id as run_id, run.filename as filename, feature.exp_RT as RT, feature.norm_RT as iRT, feature.id as id, peptide_ipf.unmodified_sequence as Sequence, peptide_ipf.modified_sequence as FullUniModPeptideName, precursor.charge as Charge, precursor.precursor_mz as mz, feature_ms2.area_intensity as Intensity, feature_ms1.area_intensity as aggr_prec_Peak_Area, feature_ms1.apex_intensity as aggr_prec_Peak_Apex, aggr_Peak_Area, aggr_Peak_Apex, aggr_Fragment_Annotation, protein.protein_accession as ProteinName, precursor.library_rt as assay_rt, feature.delta_rt as delta_rt, feature.left_width as leftWidth, feature.right_width as rightWidth, score_ms1.pep as ms1_pep, score_ms2.pep as ms2_pep, score_ipf.precursor_peakgroup_pep as precursor_pep, score_ipf.pep as ipf_pep, score_ms2.score as d_score, score_ms2.qvalue as ms2_m_score, score_ipf.qvalue as m_score from precursor join precursor_peptide_mapping on precursor.id = precursor_peptide_mapping.precursor_id join peptide on precursor_peptide_mapping.peptide_id = peptide.id inner join peptide_protein_mapping on peptide.id = peptide_protein_mapping.peptide_id join protein on peptide_protein_mapping.protein_id = protein.id join feature on feature.precursor_id = precursor.id join run on run.id = feature.run_id join feature_ms1 on feature_ms1.feature_id = feature.id join feature_ms2 on feature_ms2.feature_id = feature.id join (select feature_id, group_concat(area_intensity,';') as aggr_Peak_Area, group_concat(apex_intensity,';') as aggr_Peak_Apex, group_concat(transition_id,';') as aggr_Fragment_Annotation from feature_transition inner join transition on feature_transition.transition_id = transition.id where transition.decoy == 0 group by feature_id) as feature_transition on feature_transition.feature_id = feature.id join score_ms1 on score_ms1.feature_id = feature.id join score_ms2 on score_ms2.feature_id = feature.id join score_ipf on score_ipf.feature_id = feature.id join peptide as peptide_ipf on score_ipf.peptide_id = peptide_ipf.id;", con)
        else:
            data = pd.read_sql_query("select run.id as id_run, peptide.id as id_peptide, protein.id as id_protein, peptide_ipf.modified_sequence || '_' || precursor.id as transition_group_id, precursor.decoy as decoy, run.id as run_id, run.filename as filename, feature.exp_RT as RT, feature.norm_RT as iRT, feature.id as id, peptide_ipf.unmodified_sequence as Sequence, peptide_ipf.modified_sequence as FullUniModPeptideName, precursor.charge as Charge, precursor.precursor_mz as mz, feature_ms2.area_intensity as Intensity, feature_ms1.area_intensity as aggr_prec_Peak_Area, feature_ms1.apex_intensity as aggr_prec_Peak_Apex, protein.protein_accession as ProteinName, precursor.library_rt as assay_rt, feature.delta_rt as delta_rt, feature.left_width as leftWidth, feature.right_width as rightWidth, score_ms1.pep as ms1_pep, score_ms2.pep as ms2_pep, score_ipf.precursor_peakgroup_pep as precursor_pep, score_ipf.pep as ipf_pep, score_ms2.score as d_score, score_ms2.qvalue as ms2_m_score, score_ipf.qvalue as m_score from precursor join precursor_peptide_mapping on precursor.id = precursor_peptide_mapping.precursor_id join peptide on precursor_peptide_mapping.peptide_id = peptide.id inner join peptide_protein_mapping on peptide.id = peptide_protein_mapping.peptide_id join protein on peptide_protein_mapping.protein_id = protein.id join feature on feature.precursor_id = precursor.id join run on run.id = feature.run_id join feature_ms1 on feature_ms1.feature_id = feature.id join feature_ms2 on feature_ms2.feature_id = feature.id join score_ms1 on score_ms1.feature_id = feature.id join score_ms2 on score_ms2.feature_id = feature.id join score_ipf on score_ipf.feature_id = feature.id join peptide as peptide_ipf on score_ipf.peptide_id = peptide_ipf.id;", con)

    else:
        if transition_quantification:
            data = pd.read_sql_query("select run.id as id_run, peptide.id as id_peptide, protein.id as id_protein, precursor.id as transition_group_id, precursor.decoy as decoy, run.id as run_id, run.filename as filename, feature.exp_RT as RT, feature.norm_RT as iRT, feature.id as id, peptide.unmodified_sequence as Sequence, peptide.modified_sequence as FullUniModPeptideName, precursor.charge as Charge, precursor.precursor_mz as mz, feature_ms2.area_intensity as Intensity, feature_ms1.area_intensity as aggr_prec_Peak_Area, feature_ms1.apex_intensity as aggr_prec_Peak_Apex, aggr_Peak_Area, aggr_Peak_Apex, aggr_Fragment_Annotation, protein.protein_accession as ProteinName, precursor.library_rt as assay_rt, feature.delta_rt as delta_rt, feature.left_width as leftWidth, feature.right_width as rightWidth, score_ms2.score as d_score, score_ms2.qvalue as m_score from precursor join precursor_peptide_mapping on precursor.id = precursor_peptide_mapping.precursor_id join peptide on precursor_peptide_mapping.peptide_id = peptide.id inner join peptide_protein_mapping on peptide.id = peptide_protein_mapping.peptide_id join protein on peptide_protein_mapping.protein_id = protein.id join feature on feature.precursor_id = precursor.id join run on run.id = feature.run_id join feature_ms1 on feature_ms1.feature_id = feature.id join feature_ms2 on feature_ms2.feature_id = feature.id join (select feature_id, group_concat(area_intensity,';') as aggr_Peak_Area, group_concat(apex_intensity,';') as aggr_Peak_Apex, group_concat(transition_id,';') as aggr_Fragment_Annotation from feature_transition inner join transition on feature_transition.transition_id = transition.id group by feature_id) as feature_transition on feature_transition.feature_id = feature.id join score_ms2 on score_ms2.feature_id = feature.id;", con)
        else:
            data = pd.read_sql_query("select run.id as id_run, peptide.id as id_peptide, protein.id as id_protein, precursor.id as transition_group_id, precursor.decoy as decoy, run.id as run_id, run.filename as filename, feature.exp_RT as RT, feature.norm_RT as iRT, feature.id as id, peptide.unmodified_sequence as Sequence, peptide.modified_sequence as FullUniModPeptideName, precursor.charge as Charge, precursor.precursor_mz as mz, feature_ms2.area_intensity as Intensity, feature_ms1.area_intensity as aggr_prec_Peak_Area, feature_ms1.apex_intensity as aggr_prec_Peak_Apex, protein.protein_accession as ProteinName, precursor.library_rt as assay_rt, feature.delta_rt as delta_rt, feature.left_width as leftWidth, feature.right_width as rightWidth, score_ms2.score as d_score, score_ms2.qvalue as m_score from precursor join precursor_peptide_mapping on precursor.id = precursor_peptide_mapping.precursor_id join peptide on precursor_peptide_mapping.peptide_id = peptide.id inner join peptide_protein_mapping on peptide.id = peptide_protein_mapping.peptide_id join protein on peptide_protein_mapping.protein_id = protein.id join feature on feature.precursor_id = precursor.id join run on run.id = feature.run_id join feature_ms1 on feature_ms1.feature_id = feature.id join feature_ms2 on feature_ms2.feature_id = feature.id join score_ms2 on score_ms2.feature_id = feature.id;", con)

    peptide_present = False
    if peptide:
        peptide_present = check_sqlite_table(con, "SCORE_PEPTIDE")

    if peptide_present and peptide:
        data_peptide_run = pd.read_sql_query("select run_id as id_run, peptide_id as id_peptide, qvalue as m_score_peptide_run_specific from score_peptide where context == 'run-specific';", con)
        if len(data_peptide_run.index) > 0:
            data = pd.merge(data, data_peptide_run, how='inner', on=['id_run','id_peptide'])

        data_peptide_experiment = pd.read_sql_query("select run_id as id_run, peptide_id as id_peptide, qvalue as m_score_peptide_experiment_wide from score_peptide where context == 'experiment-wide';", con)
        if len(data_peptide_experiment.index) > 0:
            data = pd.merge(data, data_peptide_experiment, on=['id_run','id_peptide'])

        data_peptide_global = pd.read_sql_query("select peptide_id as id_peptide, qvalue as m_score_peptide_global from score_peptide where context == 'global';", con)
        if len(data_peptide_global.index) > 0:
            data = pd.merge(data, data_peptide_global[data_peptide_global['m_score_peptide_global'] < max_global_peptide_qvalue], on=['id_peptide'])

    protein_present = False
    if protein:
        protein_present = check_sqlite_table(con, "SCORE_PROTEIN")

    if protein_present and protein:
        data_protein_run = pd.read_sql_query("select run_id as id_run, protein_id as id_protein, qvalue as m_score_protein_run_specific from score_protein where context == 'run-specific';", con)
        if len(data_protein_run.index) > 0:
            data = pd.merge(data, data_protein_run, how='inner', on=['id_run','id_protein'])

        data_protein_experiment = pd.read_sql_query("select run_id as id_run, protein_id as id_protein, qvalue as m_score_protein_experiment_wide from score_protein where context == 'experiment-wide';", con)
        if len(data_protein_experiment.index) > 0:
            data = pd.merge(data, data_protein_experiment, on=['id_run','id_protein'])

        data_protein_global = pd.read_sql_query("select protein_id as id_protein, qvalue as m_score_protein_global from score_protein where context == 'global';", con)
        if len(data_protein_global.index) > 0:
            data = pd.merge(data, data_protein_global[data_protein_global['m_score_protein_global'] < max_global_protein_qvalue], on=['id_protein'])

    if outcsv:
        sep = ","
    else:
        sep = "\t"

    if format == 'legacy':
        data.drop(['id_run','id_peptide','id_protein'], axis=1).to_csv(outfile, sep=sep, index=False)
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

