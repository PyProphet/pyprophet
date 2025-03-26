from pyprophet.data_handling import check_sqlite_table
import pandas as pd
import os
import sqlite3
import click

from .report import plot_scores

def precursor_report(con, max_rs_peakgroup_qvalue):
    idx_query = '''
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_precursor_glycopeptide_mapping_precursor_id ON PRECURSOR_GLYCOPEPTIDE_MAPPING (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);

CREATE INDEX IF NOT EXISTS idx_precursor_glycopeptide_mapping_peptide_id ON PRECURSOR_GLYCOPEPTIDE_MAPPING (GLYCOPEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_glycopeptide_glycopeptide_id ON GLYCOPEPTIDE (ID);

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
    if check_sqlite_table(con, "SCORE_MS2_PART_PEPTIDE"):
        idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_part_peptide_feature_id ON SCORE_MS2_PART_PEPTIDE (FEATURE_ID);"
    if check_sqlite_table(con, "SCORE_MS2_PART_GLYCAN"):
        idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_part_glycan_feature_id ON SCORE_MS2_PART_GLYCAN (FEATURE_ID);"

    if max_rs_peakgroup_qvalue is not None:
        qvalue_filter = 'WHERE SCORE_MS2.QVALUE < %s' % max_rs_peakgroup_qvalue
    else:
        qvalue_filter = ''
        
    query = '''
SELECT RUN.ID AS id_run,
       GLYCOPEPTIDE.ID AS id_glycopeptide,
       PEPTIDE.ID AS id_peptide,
       PRECURSOR.ID AS transition_group_id,
       PRECURSOR.DECOY AS decoy,
       GLYCOPEPTIDE.DECOY_PEPTIDE AS decoy_peptide,
       GLYCOPEPTIDE.DECOY_GLYCAN AS decoy_glycan,
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
       GLYCAN.GLYCAN_STRUCT AS GlycanStruct,
       GLYCAN.GLYCAN_COMPOSITION AS GlycanComposition,
       GLYCOPEPTIDE.GLYCAN_SITE AS GlycanSite,       
       PRECURSOR.CHARGE AS Charge,
       PRECURSOR.PRECURSOR_MZ AS mz,
       FEATURE_MS2.AREA_INTENSITY AS Intensity,
       FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
       FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
       FEATURE.LEFT_WIDTH AS leftWidth,
       FEATURE.RIGHT_WIDTH AS rightWidth,
       SCORE_MS2.RANK AS peak_group_rank,
       SCORE_MS2.SCORE AS d_score,
       SCORE_MS2.PEP AS pep,
       SCORE_MS2.QVALUE AS m_score,
       SCORE_MS2_PART_PEPTIDE.SCORE AS d_score_peptide,
       SCORE_MS2_PART_PEPTIDE.PEP AS pep_peptide,
       SCORE_MS2_PART_GLYCAN.SCORE AS d_score_glycan,
       SCORE_MS2_PART_GLYCAN.PEP AS pep_glycan
FROM PRECURSOR
INNER JOIN PRECURSOR_GLYCOPEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_GLYCOPEPTIDE_MAPPING.PRECURSOR_ID
INNER JOIN GLYCOPEPTIDE ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID = GLYCOPEPTIDE.ID
INNER JOIN GLYCOPEPTIDE_PEPTIDE_MAPPING ON GLYCOPEPTIDE.ID = GLYCOPEPTIDE_PEPTIDE_MAPPING.GLYCOPEPTIDE_ID
INNER JOIN PEPTIDE ON GLYCOPEPTIDE_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
INNER JOIN GLYCOPEPTIDE_GLYCAN_MAPPING ON GLYCOPEPTIDE.ID = GLYCOPEPTIDE_GLYCAN_MAPPING.GLYCOPEPTIDE_ID
INNER JOIN GLYCAN ON GLYCOPEPTIDE_GLYCAN_MAPPING.GLYCAN_ID = GLYCAN.ID

INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
LEFT JOIN SCORE_MS2_PART_PEPTIDE ON SCORE_MS2_PART_PEPTIDE.FEATURE_ID = FEATURE.ID
LEFT JOIN SCORE_MS2_PART_GLYCAN ON SCORE_MS2_PART_GLYCAN.FEATURE_ID = FEATURE.ID

%s
ORDER BY transition_group_id,
         peak_group_rank;
''' % (qvalue_filter)
    
    con.executescript(idx_query)
    data = pd.read_sql_query(query, con)
    
    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_glycopeptide_glycosite_mapping_glycosite_id ON GLYCOPEPTIDE_GLYCOSITE_MAPPING (GLYCOSITE_ID);
CREATE INDEX IF NOT EXISTS idx_glycosite_glycosite_id ON GLYCOSITE (ID);
CREATE INDEX IF NOT EXISTS idx_glycopeptide_glycosite_mapping_glycopeptide_id ON GLYCOPEPTIDE_GLYCOSITE_MAPPING (GLYCOPEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_glycosite_protein_mapping_protein_id ON GLYCOSITE_PROTEIN_MAPPING (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_protein_protein_id ON PROTEIN (ID);
CREATE INDEX IF NOT EXISTS idx_glycosite_protein_mapping_glycosite_id ON GLYCOSITE_PROTEIN_MAPPING (GLYCOSITE_ID);
''')
    data_protein_glycosite = pd.read_sql_query('''
SELECT GLYCOPEPTIDE_ID AS id_glycopeptide,
       GROUP_CONCAT(PROTEIN.PROTEIN_ACCESSION,';') AS ProteinName,
       GROUP_CONCAT(GLYCOSITE.PROTEIN_GLYCOSITE,';') AS ProteinGlycoSite
FROM GLYCOPEPTIDE_GLYCOSITE_MAPPING
INNER JOIN GLYCOSITE ON GLYCOPEPTIDE_GLYCOSITE_MAPPING.GLYCOSITE_ID = GLYCOSITE.ID
INNER JOIN GLYCOSITE_PROTEIN_MAPPING ON GLYCOSITE.ID = GLYCOSITE_PROTEIN_MAPPING.GLYCOSITE_ID
INNER JOIN PROTEIN ON GLYCOSITE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
GROUP BY GLYCOPEPTIDE_ID;
''', con)
    data = pd.merge(data, data_protein_glycosite, how='inner', on=['id_glycopeptide'])
    
    return data


def transition_report(con, max_transition_pep):
    if max_transition_pep is not None:
        pep_filter = 'AND SCORE_TRANSITION.PEP < %s' % max_transition_pep
    else:
        pep_filter = ''
    
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
WHERE TRANSITION.DECOY == 0 %s
GROUP BY FEATURE_TRANSITION.FEATURE_ID
''' % pep_filter
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
    
    con.executescript(idx_transition_query) 
    data = pd.read_sql_query(transition_query, con)
    return data


def glycopeptide_report(con, max_global_glycopeptide_qvalue):
    if max_global_glycopeptide_qvalue is not None:
        qvalue_filter = 'AND SCORE_GLYCOPEPTIDE.QVALUE < %s' % \
            max_global_glycopeptide_qvalue
    else:
        qvalue_filter = ''
        
    data = None
    
    for context in ['run-specific', 'experiment-wide', 'global']:
        context_suffix = '_' + context.replace('-', '_')
        if context == 'global':
            run_id = ''
        else:
            run_id = 'RUN_ID AS id_run,'
            
        for part in ['peptide', 'glycan', 'total']:
            if part == 'total':
                part_suffix = ''
                table_part_suffix = ''
                m_score = ', QVALUE AS m_score_glycopeptide%s%s' % (part_suffix, context_suffix)
                m_score_filter = qvalue_filter
            else:
                part_suffix = '_' + part
                table_part_suffix = '_PART_' + part.upper()
                m_score = ''
                m_score_filter = ''
            
            if not check_sqlite_table(con, "SCORE_GLYCOPEPTIDE" + table_part_suffix):
                continue
    
            data_glycopeptide = pd.read_sql_query('''
SELECT %(run_id)s
       GLYCOPEPTIDE_ID AS id_glycopeptide,
       PEP AS pep_glycopeptide%(part_suffix)s%(context_suffix)s
       %(m_score)s
FROM SCORE_GLYCOPEPTIDE%(table_part_suffix)s
WHERE CONTEXT == '%(context)s'
      %(m_score_filter)s;
''' % {
    'run_id': run_id,
    'part_suffix': part_suffix, 
    'table_part_suffix': table_part_suffix,
    'context_suffix': context_suffix,
    'context': context,
    'm_score': m_score,
    'm_score_filter': m_score_filter
}, con)
            
            if len(data_glycopeptide.index) > 0:
                if data is None:
                    data = data_glycopeptide
                else:
                    if 'id_run' in data.columns and 'id_run' in data_glycopeptide.columns:
                        on = ['id_run', 'id_glycopeptide']
                    else:
                        on = ['id_glycopeptide']
                    data = pd.merge(data, data_glycopeptide, on=on)

    return data


def glycoform_report(con, 
                     match_precursor,
                     max_glycoform_pep,
                     max_glycoform_qvalue,
                     max_rs_peakgroup_qvalue):
    if not check_sqlite_table(con, "SCORE_GLYCOFORM"):
        raise click.ClickException("No glycoform scores.")
        
    idx_query = ''
    if check_sqlite_table(con, "FEATURE_MS1"):
        idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);"
    if check_sqlite_table(con, "SCORE_MS1"):
        idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms1_feature_id ON SCORE_MS1 (FEATURE_ID);"
        score_ms1_pep = "SCORE_MS1.PEP"
        link_ms1 = "LEFT JOIN SCORE_MS1 ON SCORE_MS1.FEATURE_ID = FEATURE.ID"
    else:
        score_ms1_pep = "NULL"
        link_ms1 = ""
    if check_sqlite_table(con, "SCORE_MS2"):
        idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);"
    if check_sqlite_table(con, "SCORE_GLYCOFORM"):
        idx_query += "CREATE INDEX IF NOT EXISTS idx_score_glycoform_feature_id ON SCORE_GLYCOFORM (FEATURE_ID);"
        idx_query += "CREATE INDEX IF NOT EXISTS idx_score_glycoform_glycopeptide_id ON SCORE_GLYCOFORM (GLYCOPEPTIDE_ID);"
        
    if match_precursor == 'exact':
        glycofrom_match_precursor = ''
        match_precursor_filter = 'GLYCOPEPTIDE.ID = GLYCOPEPTIDE_GLYCOFORM.ID'
        transition_group_id = 'PRECURSOR.ID'
    elif match_precursor == 'glycan_composition':
        glycofrom_match_precursor = ''
        match_precursor_filter = 'GLYCAN.GLYCAN_COMPOSITION = GLYCAN_GLYCOFORM.GLYCAN_COMPOSITION'
        transition_group_id = 'PRECURSOR.ID'
    else:
        glycofrom_match_precursor = 'GLYCOPEPTIDE.ID = GLYCOPEPTIDE_GLYCOFORM.ID AS glycofrom_match_precursor,'
        match_precursor_filter = '1 = 1'
        transition_group_id = '''
PRECURSOR.ID || '_' || 
PEPTIDE_GLYCOFORM.MODIFIED_SEQUENCE || '_' || 
GLYCOPEPTIDE_GLYCOFORM.GLYCAN_SITE || ',' ||
GLYCAN_GLYCOFORM.GLYCAN_STRUCT
'''

    if max_rs_peakgroup_qvalue is not None:
        ms2_qvalue_filter = 'AND SCORE_MS2.QVALUE < %s' % max_rs_peakgroup_qvalue
    else:
        ms2_qvalue_filter = ''
    if max_glycoform_pep is not None:
        glycoform_pep_filter = 'AND SCORE_GLYCOFORM.PEP < %s' % max_glycoform_pep
    else:
        glycoform_pep_filter = ''
    if max_glycoform_qvalue is not None:
        glycoform_qvalue_filter = 'AND SCORE_GLYCOFORM.QVALUE < %s' % max_glycoform_qvalue
    else:
        glycoform_qvalue_filter = ''

    query = '''
SELECT RUN.ID AS id_run,
       GLYCOPEPTIDE.ID AS id_glycopeptide,
       PEPTIDE.ID AS id_peptide,
       %(transition_group_id)s AS transition_group_id,
       PRECURSOR.DECOY AS decoy,
       GLYCOPEPTIDE.DECOY_PEPTIDE AS decoy_peptide,
       GLYCOPEPTIDE.DECOY_GLYCAN AS decoy_glycan,
       RUN.ID AS run_id,
       RUN.FILENAME AS filename,
       FEATURE.EXP_RT AS RT,
       FEATURE.EXP_RT - FEATURE.DELTA_RT AS assay_rt,
       FEATURE.DELTA_RT AS delta_rt,
       FEATURE.NORM_RT AS iRT,
       PRECURSOR.LIBRARY_RT AS assay_iRT,
       FEATURE.NORM_RT - PRECURSOR.LIBRARY_RT AS delta_iRT,
       FEATURE.ID AS id,
       PEPTIDE_GLYCOFORM.UNMODIFIED_SEQUENCE AS Sequence,
       PEPTIDE_GLYCOFORM.MODIFIED_SEQUENCE AS FullPeptideName,
       GLYCAN_GLYCOFORM.GLYCAN_STRUCT AS GlycanStruct,
       GLYCAN_GLYCOFORM.GLYCAN_COMPOSITION AS GlycanComposition,
       GLYCOPEPTIDE_GLYCOFORM.GLYCAN_SITE AS GlycanSite,
       %(glycofrom_match_precursor)s
       PRECURSOR.CHARGE AS Charge,
       PRECURSOR.PRECURSOR_MZ AS mz,
       FEATURE_MS2.AREA_INTENSITY AS Intensity,
       FEATURE_MS1.AREA_INTENSITY AS aggr_prec_Peak_Area,
       FEATURE_MS1.APEX_INTENSITY AS aggr_prec_Peak_Apex,
       FEATURE.LEFT_WIDTH AS leftWidth,
       FEATURE.RIGHT_WIDTH AS rightWidth,
       %(score_ms1_pep)s AS ms1_pep,
       SCORE_MS2.PEP AS ms2_pep,
       SCORE_GLYCOFORM.PRECURSOR_PEAKGROUP_PEP AS precursor_pep,
       SCORE_GLYCOFORM.PEP AS glycoform_pep,
       SCORE_GLYCOFORM.QVALUE AS m_score,
       SCORE_MS2.RANK AS peak_group_rank,
       SCORE_MS2.SCORE AS d_score,
       SCORE_MS2.QVALUE AS ms2_m_score,
       SCORE_MS2_PART_PEPTIDE.SCORE AS d_score_peptide,
       SCORE_MS2_PART_PEPTIDE.PEP AS ms2_pep_peptide,
       SCORE_MS2_PART_GLYCAN.SCORE AS d_score_glycan,
       SCORE_MS2_PART_GLYCAN.PEP AS ms2_pep_glycan       
FROM PRECURSOR
INNER JOIN PRECURSOR_GLYCOPEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_GLYCOPEPTIDE_MAPPING.PRECURSOR_ID
INNER JOIN GLYCOPEPTIDE ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID = GLYCOPEPTIDE.ID
INNER JOIN GLYCOPEPTIDE_PEPTIDE_MAPPING ON GLYCOPEPTIDE.ID = GLYCOPEPTIDE_PEPTIDE_MAPPING.GLYCOPEPTIDE_ID
INNER JOIN PEPTIDE ON GLYCOPEPTIDE_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
INNER JOIN GLYCOPEPTIDE_GLYCAN_MAPPING ON GLYCOPEPTIDE.ID = GLYCOPEPTIDE_GLYCAN_MAPPING.GLYCOPEPTIDE_ID
INNER JOIN GLYCAN ON GLYCOPEPTIDE_GLYCAN_MAPPING.GLYCAN_ID = GLYCAN.ID

INNER JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN RUN ON RUN.ID = FEATURE.RUN_ID
LEFT JOIN FEATURE_MS1 ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
LEFT JOIN FEATURE_MS2 ON FEATURE_MS2.FEATURE_ID = FEATURE.ID
%(link_ms1)s
LEFT JOIN SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.ID
LEFT JOIN SCORE_MS2_PART_PEPTIDE ON SCORE_MS2_PART_PEPTIDE.FEATURE_ID = FEATURE.ID
LEFT JOIN SCORE_MS2_PART_GLYCAN ON SCORE_MS2_PART_GLYCAN.FEATURE_ID = FEATURE.ID
LEFT JOIN SCORE_GLYCOFORM ON SCORE_GLYCOFORM.FEATURE_ID = FEATURE.ID

INNER JOIN GLYCOPEPTIDE AS GLYCOPEPTIDE_GLYCOFORM ON SCORE_GLYCOFORM.GLYCOPEPTIDE_ID = GLYCOPEPTIDE_GLYCOFORM.ID
INNER JOIN GLYCOPEPTIDE_PEPTIDE_MAPPING AS GLYCOPEPTIDE_PEPTIDE_MAPPING_GLYCOFORM ON GLYCOPEPTIDE_GLYCOFORM.ID = GLYCOPEPTIDE_PEPTIDE_MAPPING_GLYCOFORM.GLYCOPEPTIDE_ID
INNER JOIN PEPTIDE AS PEPTIDE_GLYCOFORM ON GLYCOPEPTIDE_PEPTIDE_MAPPING_GLYCOFORM.PEPTIDE_ID = PEPTIDE_GLYCOFORM.ID
INNER JOIN GLYCOPEPTIDE_GLYCAN_MAPPING AS GLYCOPEPTIDE_GLYCAN_MAPPING_GLYCOFORM ON GLYCOPEPTIDE_GLYCOFORM.ID = GLYCOPEPTIDE_GLYCAN_MAPPING_GLYCOFORM.GLYCOPEPTIDE_ID
INNER JOIN GLYCAN AS GLYCAN_GLYCOFORM ON GLYCOPEPTIDE_GLYCAN_MAPPING_GLYCOFORM.GLYCAN_ID = GLYCAN_GLYCOFORM.ID

WHERE %(match_precursor_filter)s
      %(ms2_qvalue_filter)s
      %(glycoform_pep_filter)s
      %(glycoform_qvalue_filter)s
ORDER BY transition_group_id,
         peak_group_rank;
''' % {
    'transition_group_id': transition_group_id, 
    'glycofrom_match_precursor': glycofrom_match_precursor,
    'score_ms1_pep': score_ms1_pep, 
    'link_ms1': link_ms1, 
    'match_precursor_filter': match_precursor_filter,
    'ms2_qvalue_filter': ms2_qvalue_filter, 
    'glycoform_pep_filter': glycoform_pep_filter, 
    'glycoform_qvalue_filter': glycoform_qvalue_filter
}

    con.executescript(idx_query) 
    data = pd.read_sql_query(query, con)
    
    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_glycopeptide_glycosite_mapping_glycosite_id ON GLYCOPEPTIDE_GLYCOSITE_MAPPING (GLYCOSITE_ID);
CREATE INDEX IF NOT EXISTS idx_glycosite_glycosite_id ON GLYCOSITE (ID);
CREATE INDEX IF NOT EXISTS idx_glycopeptide_glycosite_mapping_glycopeptide_id ON GLYCOPEPTIDE_GLYCOSITE_MAPPING (GLYCOPEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_glycosite_protein_mapping_protein_id ON GLYCOSITE_PROTEIN_MAPPING (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_protein_protein_id ON PROTEIN (ID);
CREATE INDEX IF NOT EXISTS idx_glycosite_protein_mapping_glycosite_id ON GLYCOSITE_PROTEIN_MAPPING (GLYCOSITE_ID);
''')
    data_protein_glycosite = pd.read_sql_query('''
SELECT GLYCOPEPTIDE_ID AS id_glycopeptide,
       GROUP_CONCAT(PROTEIN.PROTEIN_ACCESSION,';') AS ProteinName,
       GROUP_CONCAT(GLYCOSITE.PROTEIN_GLYCOSITE,';') AS ProteinGlycoSite
FROM GLYCOPEPTIDE_GLYCOSITE_MAPPING
INNER JOIN GLYCOSITE ON GLYCOPEPTIDE_GLYCOSITE_MAPPING.GLYCOSITE_ID = GLYCOSITE.ID
INNER JOIN GLYCOSITE_PROTEIN_MAPPING ON GLYCOSITE.ID = GLYCOSITE_PROTEIN_MAPPING.GLYCOSITE_ID
INNER JOIN PROTEIN ON GLYCOSITE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
GROUP BY GLYCOPEPTIDE_ID;
''', con)
    data = pd.merge(data, data_protein_glycosite, how='inner', on=['id_glycopeptide'])
    
    return data


def export_tsv(infile, outfile, format='legacy_merged', outcsv=False, 
               transition_quantification=True, max_transition_pep=0.7, 
               glycoform=False, glycoform_match_precursor='glycan_composition',
               max_glycoform_pep=None,
               max_glycoform_qvalue=0.01,
               max_rs_peakgroup_qvalue=0.05,
               glycopeptide=True, max_global_glycopeptide_qvalue=0.01):
    
    osw = sqlite3.connect(infile)
    
    click.echo("Info: Reading peak group-level results.")
    if not glycoform:
        data = precursor_report(
            osw, 
            max_rs_peakgroup_qvalue=max_rs_peakgroup_qvalue
        )
    else:
        data = glycoform_report(
            osw, 
            match_precursor=glycoform_match_precursor,
            max_glycoform_pep=max_glycoform_pep,
            max_glycoform_qvalue=max_glycoform_qvalue,
            max_rs_peakgroup_qvalue=max_rs_peakgroup_qvalue
        )
            
    if transition_quantification:
        click.echo("Info: Reading transition-level results.")
        data_transition = transition_report(
            osw, 
            max_transition_pep=max_transition_pep
        )
        if data_transition is not None and len(data_transition.index) > 0:
            data = pd.merge(data, data_transition, how='left', on=['id'])
        
    if glycopeptide:
        click.echo("Info: Reading glycopeptide-level results.")
        data_glycopeptide = glycopeptide_report(
            osw,
            max_global_glycopeptide_qvalue=max_global_glycopeptide_qvalue
        )        
        if data_glycopeptide is not None and len(data_glycopeptide.index) > 0:
            if 'id_run' in data_glycopeptide.columns:
                data = pd.merge(data, data_glycopeptide, how='inner', on=['id_run','id_glycopeptide'])
            else:
                data = pd.merge(data, data_glycopeptide, how='inner', on=['id_glycopeptide'])
        
    if outcsv:
        sep = ","
    else:
        sep = "\t"

    if format == 'legacy_split':
        data = data.drop(['id_run', 'id_glycopeptide', 'id_peptide'], axis=1)
        data.groupby('filename').apply(lambda x: x.to_csv(
            os.path.basename(x['filename'].values[0]) + '.tsv', 
            sep=sep, index=False
        ))
    elif format == 'legacy_merged':
        data.drop(['id_run', 'id_glycopeptide', 'id_peptide'], axis=1) \
            .to_csv(outfile, sep=sep, index=False)
    elif format == 'matrix':
        data = data.iloc[data.groupby(['run_id', 'transition_group_id']) \
            .apply(lambda x: x['m_score'].idxmin())]
        
        data = data[['transition_group_id', 
                     'decoy', 'decoy_peptide', 'decoy_glycan',
                     'Sequence', 'FullPeptideName',
                     'GlycanStruct', 'GlycanComposition', 'GlycanSite',
                     'Charge',
                     'ProteinName', 'ProteinGlycoSite', 'filename', 'Intensity']]
        
        data = data.pivot_table(
            index=list(data.columns \
                .difference(['filename', 'Intensity'], sort=False)), 
            columns='filename', values='Intensity'
        )
        data.to_csv(outfile, sep=sep, index=True)

    osw.close()
    
    
def export_score_plots(infile):
    con = sqlite3.connect(infile)

    if check_sqlite_table(con, "SCORE_MS2") and \
        check_sqlite_table(con, "SCORE_MS2_PART_PEPTIDE") and \
        check_sqlite_table(con, "SCORE_MS2_PART_GLYCAN"):
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
           DECOY_PEPTIDE,
           DECOY_GLYCAN
    FROM PRECURSOR_GLYCOPEPTIDE_MAPPING
    INNER JOIN GLYCOPEPTIDE 
    ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID == GLYCOPEPTIDE.ID) AS DECOY 
   ON FEATURE.PRECURSOR_ID = DECOY.ID
   
INNER JOIN
  (SELECT PRECURSOR_ID AS ID,
          COUNT(*) AS VAR_TRANSITION_NUM_SCORE
   FROM TRANSITION_PRECURSOR_MAPPING
   INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
   WHERE DETECTING==1
   GROUP BY PRECURSOR_ID) AS VAR_TRANSITION_SCORE ON FEATURE.PRECURSOR_ID = VAR_TRANSITION_SCORE.ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID

INNER JOIN
  (SELECT FEATURE_ID,
          SCORE AS SCORE_PEPTIDE          
   FROM SCORE_MS2_PART_PEPTIDE) AS SCORE_MS2_PART_PEPTIDE 
  ON FEATURE.ID = SCORE_MS2_PART_PEPTIDE.FEATURE_ID
  
INNER JOIN
  (SELECT FEATURE_ID,
          SCORE AS SCORE_GLYCAN          
   FROM SCORE_MS2_PART_GLYCAN) AS SCORE_MS2_PART_GLYCAN
  ON FEATURE.ID = SCORE_MS2_PART_GLYCAN.FEATURE_ID
  
WHERE RANK == 1
ORDER BY RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
''', con)
        plot_scores(table_ms2, outfile, title=infile + ': MS2 scores')

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
  
INNER JOIN
   (SELECT PRECURSOR_ID AS ID,
           DECOY_PEPTIDE,
           DECOY_GLYCAN
    FROM PRECURSOR_GLYCOPEPTIDE_MAPPING
    INNER JOIN GLYCOPEPTIDE 
    ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID == GLYCOPEPTIDE.ID) AS DECOY 
   ON FEATURE.PRECURSOR_ID = DECOY.ID
   
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
WHERE RANK == 1
ORDER BY RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
''', con)
        plot_scores(table_ms1, outfile, title=infile + ': MS1 scores')

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
        plot_scores(table_transition, outfile, title=infile + ': transition scores')

    con.close()
    
    