import pandas as pd
import numpy as np
import sqlite3
import click
import os

from .data_handling import check_sqlite_table
from .report import plot_scores


def export_tsv(infile, outfile, format, outcsv, transition_quantification, max_transition_pep, ipf, ipf_max_peptidoform_pep, max_rs_peakgroup_qvalue, peptide, max_global_peptide_qvalue, protein, max_global_protein_qvalue):

    con = sqlite3.connect(infile)

    ipf_present = False
    if ipf:
        ipf_present = check_sqlite_table(con, "SCORE_IPF")

    # Main query for peptidoform IPF
    if ipf_present and ipf=='peptidoform':
        idx_query = '''
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);

CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);

CREATE INDEX IF NOT EXISTS idx_run_run_id ON RUN (ID);
CREATE INDEX IF NOT EXISTS idx_feature_run_id ON FEATURE (RUN_ID);

CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
'''
        if check_sqlite_table(con, "FEATURE_MS1"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);"
        if check_sqlite_table(con, "FEATURE_MS2"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);"
        if check_sqlite_table(con, "SCORE_MS1"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms1_feature_id ON SCORE_MS1 (FEATURE_ID);"
          score_ms1_pep = "SCORE_MS1.PEP"
          link_ms1 = "LEFT JOIN SCORE_MS1 ON SCORE_MS1.FEATURE_ID = FEATURE.ID"
        else:
          score_ms1_pep = "NULL"
          link_ms1 = ""
        if check_sqlite_table(con, "SCORE_MS2"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);"
        if check_sqlite_table(con, "SCORE_IPF"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ipf_feature_id ON SCORE_IPF (FEATURE_ID);"
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ipf_peptide_id ON SCORE_IPF (PEPTIDE_ID);"

        query = '''
SELECT RUN.ID AS id_run,
       PEPTIDE.ID AS id_peptide,
       PEPTIDE_IPF.MODIFIED_SEQUENCE || '_' || PRECURSOR.ID AS transition_group_id,
       PRECURSOR.DECOY AS decoy,
       RUN.ID AS run_id,
       RUN.FILENAME AS filename,
       FEATURE.EXP_RT AS RT,
       FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
       FEATURE.DELTA_RT AS delta_rt,
       FEATURE.NORM_RT AS iRT,
       PRECURSOR.LIBRARY_RT AS assay_iRT,
       FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT,
       FEATURE.ID AS id,
       PEPTIDE_IPF.UNMODIFIED_SEQUENCE AS Sequence,
       PEPTIDE_IPF.MODIFIED_SEQUENCE AS FullPeptideName,
       PRECURSOR.CHARGE AS Charge,
       PRECURSOR.PRECURSOR_MZ AS mz,
       FEATURE_MS2.AREA_INTENSITY AS Intensity,
       FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
       FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
       FEATURE.LEFT_WIDTH AS leftWidth,
       FEATURE.RIGHT_WIDTH AS rightWidth,
       %s AS ms1_pep,
       SCORE_MS2.PEP AS ms2_pep,
       SCORE_IPF.PRECURSOR_PEAKGROUP_PEP AS precursor_pep,
       SCORE_IPF.PEP AS ipf_pep,
       SCORE_MS2.RANK AS peak_group_rank,
       SCORE_MS2.SCORE AS d_score,
       SCORE_MS2.QVALUE AS ms2_m_score,
       SCORE_IPF.QVALUE AS m_score
FROM PRECURSOR
INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
%s
LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
LEFT JOIN SCORE_IPF ON SCORE_IPF.FEATURE_ID = FEATURE.ID
INNER JOIN PEPTIDE AS PEPTIDE_IPF ON SCORE_IPF.PEPTIDE_ID = PEPTIDE_IPF.ID
WHERE SCORE_MS2.QVALUE < %s AND SCORE_IPF.PEP < %s
ORDER BY transition_group_id,
         peak_group_rank;
''' % (score_ms1_pep, link_ms1, max_rs_peakgroup_qvalue, ipf_max_peptidoform_pep)
    # Main query for augmented IPF
    elif ipf_present and ipf=='augmented':
        idx_query = '''
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);

CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);

CREATE INDEX IF NOT EXISTS idx_run_run_id ON RUN (ID);
CREATE INDEX IF NOT EXISTS idx_feature_run_id ON FEATURE (RUN_ID);

CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
'''
        if check_sqlite_table(con, "FEATURE_MS1"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);"
        if check_sqlite_table(con, "FEATURE_MS2"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);"
        if check_sqlite_table(con, "SCORE_MS1"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms1_feature_id ON SCORE_MS1 (FEATURE_ID);"
          score_ms1_pep = "SCORE_MS1.PEP"
          link_ms1 = "LEFT JOIN SCORE_MS1 ON SCORE_MS1.FEATURE_ID = FEATURE.ID"
        else:
          score_ms1_pep = "NULL"
          link_ms1 = ""
        if check_sqlite_table(con, "SCORE_MS2"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);"
        if check_sqlite_table(con, "SCORE_IPF"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ipf_feature_id ON SCORE_IPF (FEATURE_ID);"
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ipf_peptide_id ON SCORE_IPF (PEPTIDE_ID);"

        query = '''
SELECT RUN.ID AS id_run,
       PEPTIDE.ID AS id_peptide,
       PRECURSOR.ID AS transition_group_id,
       PRECURSOR.DECOY AS decoy,
       RUN.ID AS run_id,
       RUN.FILENAME AS filename,
       FEATURE.EXP_RT AS RT,
       FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
       FEATURE.DELTA_RT AS delta_rt,
       FEATURE.NORM_RT AS iRT,
       PRECURSOR.LIBRARY_RT AS assay_iRT,
       FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT,
       FEATURE.ID AS id,
       PEPTIDE.UNMODIFIED_SEQUENCE AS Sequence,
       PEPTIDE.MODIFIED_SEQUENCE AS FullPeptideName,
       PRECURSOR.CHARGE AS Charge,
       PRECURSOR.PRECURSOR_MZ AS mz,
       FEATURE_MS2.AREA_INTENSITY AS Intensity,
       FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
       FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
       FEATURE.LEFT_WIDTH AS leftWidth,
       FEATURE.RIGHT_WIDTH AS rightWidth,
       SCORE_MS2.RANK AS peak_group_rank,
       SCORE_MS2.SCORE AS d_score,
       SCORE_MS2.QVALUE AS m_score,
       %s AS ms1_pep,
       SCORE_MS2.PEP AS ms2_pep
FROM PRECURSOR
INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
%s
LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
WHERE SCORE_MS2.QVALUE < %s
ORDER BY transition_group_id,
         peak_group_rank;
''' % (score_ms1_pep, link_ms1, max_rs_peakgroup_qvalue)
        query_augmented = '''
SELECT FEATURE_ID AS id,
       MODIFIED_SEQUENCE AS ipf_FullUniModPeptideName,
       PRECURSOR_PEAKGROUP_PEP AS ipf_precursor_peakgroup_pep,
       PEP AS ipf_peptidoform_pep,
       QVALUE AS ipf_peptidoform_m_score
FROM SCORE_IPF
INNER JOIN PEPTIDE ON SCORE_IPF.PEPTIDE_ID = PEPTIDE.ID
WHERE SCORE_IPF.PEP < %s;
''' % ipf_max_peptidoform_pep
	# Main query for standard OpenSWATH
    else:
        idx_query = '''
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);

CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);

CREATE INDEX IF NOT EXISTS idx_run_run_id ON RUN (ID);
CREATE INDEX IF NOT EXISTS idx_feature_run_id ON FEATURE (RUN_ID);

CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
'''
        if check_sqlite_table(con, "FEATURE_MS1"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);"
        if check_sqlite_table(con, "FEATURE_MS2"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);"
        if check_sqlite_table(con, "SCORE_MS2"):
          idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);"

        query = '''
SELECT RUN.ID AS id_run,
       PEPTIDE.ID AS id_peptide,
       PRECURSOR.ID AS transition_group_id,
       PRECURSOR.DECOY AS decoy,
       RUN.ID AS run_id,
       RUN.FILENAME AS filename,
       FEATURE.EXP_RT AS RT,
       FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
       FEATURE.DELTA_RT AS delta_rt,
       FEATURE.NORM_RT AS iRT,
       PRECURSOR.LIBRARY_RT AS assay_iRT,
       FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT,
       FEATURE.ID AS id,
       PEPTIDE.UNMODIFIED_SEQUENCE AS Sequence,
       PEPTIDE.MODIFIED_SEQUENCE AS FullPeptideName,
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
INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
WHERE SCORE_MS2.QVALUE < %s
ORDER BY transition_group_id,
         peak_group_rank;
''' % max_rs_peakgroup_qvalue

    # Execute main SQLite query
    click.echo("Info: Reading peak group-level results.")
    con.executescript(idx_query) # Add indices
    data = pd.read_sql_query(query, con)

    # Augment OpenSWATH results with IPF scores
    if ipf_present and ipf=='augmented':
      data_augmented = pd.read_sql_query(query_augmented, con)

      data_augmented = data_augmented.groupby('id').apply(lambda x: pd.Series({'ipf_FullUniModPeptideName': ";".join(x[x['ipf_peptidoform_pep'] == np.min(x['ipf_peptidoform_pep'])]['ipf_FullUniModPeptideName']), 'ipf_precursor_peakgroup_pep': x[x['ipf_peptidoform_pep'] == np.min(x['ipf_peptidoform_pep'])]['ipf_precursor_peakgroup_pep'].values[0], 'ipf_peptidoform_pep': x[x['ipf_peptidoform_pep'] == np.min(x['ipf_peptidoform_pep'])]['ipf_peptidoform_pep'].values[0], 'ipf_peptidoform_m_score': x[x['ipf_peptidoform_pep'] == np.min(x['ipf_peptidoform_pep'])]['ipf_peptidoform_m_score'].values[0]})).reset_index(level='id')

      data = pd.merge(data, data_augmented, how='left', on='id')

    # Append transition-level quantities
    if transition_quantification:
        if check_sqlite_table(con, "SCORE_TRANSITION"):
            idx_transition_query = '''
CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);
CREATE INDEX IF NOT EXISTS idx_transition_transition_id ON TRANSITION (ID);

CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id_feature_id ON FEATURE_TRANSITION (TRANSITION_ID, FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id_feature_id ON SCORE_TRANSITION (TRANSITION_ID, FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID);
'''
            transition_query = '''
SELECT FEATURE_TRANSITION.FEATURE_ID AS id,
      GROUP_CONCAT(AREA_INTENSITY,';') AS aggr_Peak_Area,
      GROUP_CONCAT(APEX_INTENSITY,';') AS aggr_Peak_Apex,
      GROUP_CONCAT(TRANSITION.ID || "_" || TRANSITION.TYPE || TRANSITION.ORDINAL || "_" || TRANSITION.CHARGE,';') AS aggr_Fragment_Annotation
FROM FEATURE_TRANSITION
INNER JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN SCORE_TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = SCORE_TRANSITION.TRANSITION_ID AND FEATURE_TRANSITION.FEATURE_ID = SCORE_TRANSITION.FEATURE_ID
WHERE TRANSITION.DECOY == 0 AND SCORE_TRANSITION.PEP < %s
GROUP BY FEATURE_TRANSITION.FEATURE_ID
''' % max_transition_pep
        else:
            idx_transition_query = '''
CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);
CREATE INDEX IF NOT EXISTS idx_transition_transition_id ON TRANSITION (ID);

CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID);
'''
            transition_query = '''
SELECT FEATURE_ID AS id,
      GROUP_CONCAT(AREA_INTENSITY,';') AS aggr_Peak_Area,
      GROUP_CONCAT(APEX_INTENSITY,';') AS aggr_Peak_Apex,
      GROUP_CONCAT(TRANSITION.ID || "_" || TRANSITION.TYPE || TRANSITION.ORDINAL || "_" || TRANSITION.CHARGE,';') AS aggr_Fragment_Annotation
FROM FEATURE_TRANSITION
INNER JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
GROUP BY FEATURE_ID
'''
        click.echo("Info: Reading transition-level results.")
        con.executescript(idx_transition_query) # Add indices
        data_transition = pd.read_sql_query(transition_query, con)
        data = pd.merge(data, data_transition, how='left', on=['id'])

    # Append concatenated protein identifier
    click.echo("Info: Reading protein identifiers.")
    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_protein_id ON PEPTIDE_PROTEIN_MAPPING (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_protein_protein_id ON PROTEIN (ID);

CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_peptide_id ON PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID);
''')
    data_protein = pd.read_sql_query('''
SELECT PEPTIDE_ID AS id_peptide,
       GROUP_CONCAT(PROTEIN.PROTEIN_ACCESSION,';') AS ProteinName
FROM PEPTIDE_PROTEIN_MAPPING
INNER JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
GROUP BY PEPTIDE_ID;
''', con)
    data = pd.merge(data, data_protein, how='inner', on=['id_peptide'])

    # Append peptide error-rate control
    peptide_present = False
    if peptide:
        peptide_present = check_sqlite_table(con, "SCORE_PEPTIDE")

    if peptide_present and peptide:
        click.echo("Info: Reading peptide-level results.")
        data_peptide_run = pd.read_sql_query('''
SELECT RUN_ID AS id_run,
       PEPTIDE_ID AS id_peptide,
       QVALUE AS m_score_peptide_run_specific
FROM SCORE_PEPTIDE
WHERE CONTEXT == 'run-specific';
''', con)
        if len(data_peptide_run.index) > 0:
            data = pd.merge(data, data_peptide_run, how='inner', on=['id_run','id_peptide'])

        data_peptide_experiment = pd.read_sql_query('''
SELECT RUN_ID AS id_run,
       PEPTIDE_ID AS id_peptide,
       QVALUE AS m_score_peptide_experiment_wide
FROM SCORE_PEPTIDE
WHERE CONTEXT == 'experiment-wide';
''', con)
        if len(data_peptide_experiment.index) > 0:
            data = pd.merge(data, data_peptide_experiment, on=['id_run','id_peptide'])

        data_peptide_global = pd.read_sql_query('''
SELECT PEPTIDE_ID AS id_peptide,
       QVALUE AS m_score_peptide_global
FROM SCORE_PEPTIDE
WHERE CONTEXT == 'global';
''', con)
        if len(data_peptide_global.index) > 0:
            data = pd.merge(data, data_peptide_global[data_peptide_global['m_score_peptide_global'] < max_global_peptide_qvalue], on=['id_peptide'])

    # Append protein error-rate control
    protein_present = False
    if protein:
        protein_present = check_sqlite_table(con, "SCORE_PROTEIN")

    if protein_present and protein:
        click.echo("Info: Reading protein-level results.")
        con.executescript('''
CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_protein_id ON PEPTIDE_PROTEIN_MAPPING (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_peptide_id ON PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_score_protein_protein_id ON SCORE_PROTEIN (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_score_protein_run_id ON SCORE_PROTEIN (RUN_ID);
''')
        data_protein_run = pd.read_sql_query('''
SELECT RUN_ID AS id_run,
       PEPTIDE_ID AS id_peptide,
       MIN(QVALUE) AS m_score_protein_run_specific
FROM PEPTIDE_PROTEIN_MAPPING
INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID
WHERE CONTEXT == 'run-specific'
GROUP BY RUN_ID,
         PEPTIDE_ID;
''', con)
        if len(data_protein_run.index) > 0:
            data = pd.merge(data, data_protein_run, how='inner', on=['id_run','id_peptide'])

        con.executescript('''
CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_protein_id ON PEPTIDE_PROTEIN_MAPPING (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_peptide_id ON PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_score_protein_protein_id ON SCORE_PROTEIN (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_score_protein_run_id ON SCORE_PROTEIN (RUN_ID);
''')
        data_protein_experiment = pd.read_sql_query('''
SELECT RUN_ID AS id_run,
       PEPTIDE_ID AS id_peptide,
       MIN(QVALUE) AS m_score_protein_experiment_wide
FROM PEPTIDE_PROTEIN_MAPPING
INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID
WHERE CONTEXT == 'experiment-wide'
GROUP BY RUN_ID,
         PEPTIDE_ID;
''', con)
        if len(data_protein_experiment.index) > 0:
            data = pd.merge(data, data_protein_experiment, how='inner', on=['id_run','id_peptide'])

        con.executescript('''
CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_protein_id ON PEPTIDE_PROTEIN_MAPPING (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_peptide_id ON PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_score_protein_protein_id ON SCORE_PROTEIN (PROTEIN_ID);
''')
        data_protein_global = pd.read_sql_query('''
SELECT PEPTIDE_ID AS id_peptide,
       MIN(QVALUE) AS m_score_protein_global
FROM PEPTIDE_PROTEIN_MAPPING
INNER JOIN SCORE_PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = SCORE_PROTEIN.PROTEIN_ID
WHERE CONTEXT == 'global'
GROUP BY PEPTIDE_ID;
''', con)
        if len(data_protein_global.index) > 0:
            data = pd.merge(data, data_protein_global[data_protein_global['m_score_protein_global'] < max_global_protein_qvalue], how='inner', on=['id_peptide'])

    if outcsv:
        sep = ","
    else:
        sep = "\t"

    if format == 'legacy_split':
        data = data.drop(['id_run','id_peptide'], axis=1)
        data.groupby('filename').apply(lambda x: x.to_csv(os.path.basename(x['filename'].values[0]) + '.tsv', sep=sep, index=False))
    elif format == 'legacy_merged':
        data.drop(['id_run','id_peptide'], axis=1).to_csv(outfile, sep=sep, index=False)
    elif format == 'matrix':
        # select top ranking peak group only
        data = data.iloc[data.groupby(['run_id','transition_group_id']).apply(lambda x: x['m_score'].idxmin())]
        # restructure dataframe to matrix
        data = data[['transition_group_id','Sequence','FullPeptideName','ProteinName','filename','Intensity']]
        data = data.pivot_table(index=['transition_group_id','Sequence','FullPeptideName','ProteinName'], columns='filename', values='Intensity')
        data.to_csv(outfile, sep=sep, index=True)

    con.close()


def export_score_plots(infile):

    con = sqlite3.connect(infile)

    if check_sqlite_table(con, "SCORE_MS2"):
        outfile = infile.split(".osw")[0] + "_ms2_score_plots.pdf"
        table_ms2 = pd.read_sql_query('''
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
''', con)
        plot_scores(table_ms2, outfile)

    if check_sqlite_table(con, "SCORE_MS1"):
        outfile = infile.split(".osw")[0] + "_ms1_score_plots.pdf"
        table_ms1 = pd.read_sql_query('''
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
''', con)
        plot_scores(table_ms1, outfile)

    if check_sqlite_table(con, "SCORE_TRANSITION"):
        outfile = infile.split(".osw")[0] + "_transition_score_plots.pdf"
        table_transition = pd.read_sql_query('''
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
''', con)
        plot_scores(table_transition, outfile)

    con.close()
