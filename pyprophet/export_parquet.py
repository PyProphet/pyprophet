import duckdb
import sqlite3
import pandas as pd
from pyprophet.export import check_sqlite_table

def getPeptideProteinScoreTable(conndb, level):
    if level == 'peptide':
        id = 'PEPTIDE_ID'
        score_table = 'SCORE_PEPTIDE'
    else: # level == 'protein'
        id = 'PROTEIN_ID'
        score_table = 'SCORE_PROTEIN'
        
    nonGlobal= conndb.sql(f"select * from {score_table} where context != 'global'").df()
    nonGlobal= nonGlobal.pivot(index=[id, 'RUN_ID'], columns='CONTEXT')
    nonGlobal.columns = [f'{score_table}.' + col.upper() for col in nonGlobal.columns.map('_'.join)]
    nonGlobal= nonGlobal.reset_index()
        
    glob = conndb.sql(f"select {id}, SCORE, PVALUE, QVALUE, PEP from {score_table} where context == 'global'").df()
    glob.columns = [f'{score_table}.' + col.upper() + '_GLOBAL' if col != id else col for col in glob.columns ]

    return nonGlobal.merge(glob, how='outer')
    
# this method is only currently supported for combined output and not with ipf
def export_to_parquet(infile, outfile, transitionLevel):
    '''
    Convert an OSW sqlite file to Parquet format

    Parameters:
        infile: (str) path to osw sqlite file
        outfile: (str) path to write out parquet file
        transitionLevel: (bool) append transition level data
        separate_runs: (bool) if the input file is a merged file, create separate parquet files
        chunksize: (int) read in the data into chunks for low-memory requirements
        threads: (int) number of threads to use for parallelizing on precursor ids
    
    Return:
        None
    '''
    condb = duckdb.connect(infile)
    con = sqlite3.connect(infile)

    # Main query for standard OpenSWATH
    idx_query = '''
    CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
    CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);
    CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);

    CREATE INDEX IF NOT EXISTS idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);
    CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID);

    CREATE INDEX IF NOT EXISTS idx_run_run_id ON RUN (ID);
    CREATE INDEX IF NOT EXISTS idx_feature_run_id ON FEATURE (RUN_ID);

    CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);

    CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_protein_id ON PEPTIDE_PROTEIN_MAPPING (PROTEIN_ID);
    CREATE INDEX IF NOT EXISTS idx_protein_protein_id ON PROTEIN (ID);
    CREATE INDEX IF NOT EXISTS idx_peptide_protein_mapping_peptide_id ON PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID);

    '''

    if check_sqlite_table(con, "FEATURE_MS1"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);"
    if check_sqlite_table(con, "FEATURE_MS2"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);"
    if check_sqlite_table(con, "SCORE_MS2"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);"
    if check_sqlite_table(con, "SCORE_PEPTIDE"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_peptide_peptide_id ON SCORE_PEPTIDE (PEPTIDE_ID);"
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_peptide_run_id ON SCORE_PEPTIDE (RUN_ID);"
    if check_sqlite_table(con, "SCORE_PROTEIN"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_protein_protein_id ON SCORE_PROTEIN (PROTEIN_ID);"
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_protein_run_id ON SCORE_PROTEIN (RUN_ID);"
    if check_sqlite_table(con, "SCORE_GENE"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_gene_gene_id ON SCORE_GENE (GENE_ID);"
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_gene_run_id ON SCORE_GENE (RUN_ID);"
    


    con.executescript(idx_query) # Add indices

    # create transition indicies (if needed)
    if transitionLevel:
        idx_transition_query = '''
        CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);
        CREATE INDEX IF NOT EXISTS idx_transition_transition_id ON TRANSITION (ID);
        CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id_feature_id ON FEATURE_TRANSITION (TRANSITION_ID, FEATURE_ID);
        CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID); '''

        con.executescript(idx_transition_query)

    
    # since do not want all of the columns (some columns are twice per table) manually select the columns want in a list, (note do not want decoy)
    # note TRAML_ID for precursor and transition are not the same
    columns = {}
    ## library 
    columns['PRECURSOR'] = ['TRAML_ID', 'GROUP_LABEL', 'PRECURSOR_MZ', 'CHARGE', 'LIBRARY_INTENSITY', 'LIBRARY_RT', 'LIBRARY_DRIFT_TIME', 'DECOY']
    columns['PEPTIDE'] = ['UNMODIFIED_SEQUENCE', 'MODIFIED_SEQUENCE']
    columns['PROTEIN'] = ['PROTEIN_ACCESSION']
    if check_sqlite_table(con, "GENE") and pd.read_sql('SELECT GENE_NAME FROM GENE', con).GENE_NAME[0]!='NA':
        columns['GENE'] = ['GENE_NAME']

    ## features
    columns['FEATURE'] = ['RUN_ID', 'EXP_RT', 'EXP_IM', 'NORM_RT', 'DELTA_RT', 'LEFT_WIDTH', 'RIGHT_WIDTH']
    columns['FEATURE_MS2'] = ['FEATURE_ID', 'AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI', 'VAR_BSERIES_SCORE', 'VAR_DOTPROD_SCORE', 'VAR_INTENSITY_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE', 'VAR_LIBRARY_CORR', 'VAR_LIBRARY_DOTPROD', 'VAR_LIBRARY_MANHATTAN', 'VAR_LIBRARY_RMSD', 'VAR_LIBRARY_ROOTMEANSQUARE', 'VAR_LIBRARY_SANGLE', 'VAR_LOG_SN_SCORE', 'VAR_MANHATTAN_SCORE', 'VAR_MASSDEV_SCORE', 'VAR_MASSDEV_SCORE_WEIGHTED', 'VAR_MI_SCORE', 'VAR_MI_WEIGHTED_SCORE', 'VAR_MI_RATIO_SCORE', 'VAR_NORM_RT_SCORE', 'VAR_XCORR_COELUTION', 'VAR_XCORR_COELUTION_WEIGHTED', 'VAR_XCORR_SHAPE', 'VAR_XCORR_SHAPE_WEIGHTED', 'VAR_YSERIES_SCORE', 'VAR_ELUTION_MODEL_FIT_SCORE', 'VAR_IM_XCORR_SHAPE', 'VAR_IM_XCORR_COELUTION', 'VAR_IM_DELTA_SCORE', 'VAR_SONAR_LAG', 'VAR_SONAR_SHAPE', 'VAR_SONAR_LOG_SN', 'VAR_SONAR_LOG_DIFF', 'VAR_SONAR_LOG_TREND', 'VAR_SONAR_RSQ']
    columns['FEATURE_MS1'] = ['APEX_INTENSITY', 'VAR_MASSDEV_SCORE', 'VAR_MI_SCORE', 'VAR_MI_CONTRAST_SCORE', 'VAR_MI_COMBINED_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE', 'VAR_IM_MS1_DELTA_SCORE', 'VAR_XCORR_COELUTION', 'VAR_XCORR_COELUTION_CONTRAST', 'VAR_XCORR_COELUTION_COMBINED', 'VAR_XCORR_SHAPE', 'VAR_XCORR_SHAPE_CONTRAST', 'VAR_XCORR_SHAPE_COMBINED']

    # check if IM columns exist
    query = con.execute("select count(*) as cntrec from pragma_table_info('feature_MS2') where name='EXP_IM'")
    hasIm = query.fetchone()[0] > 0
    if hasIm:
        imColumns = ['EXP_IM', 'DELTA_IM']
        columns['FEATURE_MS2'] = columns['FEATURE_MS2'] + imColumns
        columns['FEATURE_MS1'] = columns['FEATURE_MS1'] + imColumns

    ### pyprophet scores 
    columns['SCORE_MS2'] = ["SCORE", "RANK", "PVALUE", "QVALUE", "PEP"]

    ## other
    columns['RUN'] = ['FILENAME']


    ## mappings
    columns['PRECURSOR_PEPTIDE_MAPPING'] = ['PEPTIDE_ID', 'PRECURSOR_ID']
    #columns['TRANSITION_PRECURSOR_MAPPING'] = ['PRECURSOR_ID']
    columns['PEPTIDE_PROTEIN_MAPPING'] = ['PROTEIN_ID']
    if check_sqlite_table(con, "GENE") and pd.read_sql('SELECT GENE_NAME FROM GENE', con).GENE_NAME[0]!='NA':
        columns['PEPTIDE_GENE_MAPPING'] = ['GENE_ID']

    # transition level
    if transitionLevel:
        columns['FEATURE_TRANSITION'] = ['TRANSITION_ID', 'AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI', 'VAR_INTENSITY_SCORE', 'VAR_INTENSITY_RATIO_SCORE', 'VAR_LOG_INTENSITY', 'VAR_XCORR_COELUTION', 'VAR_XCORR_SHAPE', 'VAR_LOG_SN_SCORE', 'VAR_MASSDEV_SCORE', 'VAR_MI_SCORE', 'VAR_MI_RATIO_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE']
        columns['TRANSITION'] = ['TRAML_ID', 'PRODUCT_MZ', 'CHARGE', 'TYPE', 'ORDINAL', 'DETECTING', 'IDENTIFYING', 'QUANTIFYING', 'LIBRARY_INTENSITY']

    ### rename column names that are in common 
    whitelist = set(['PEPTIDE_ID', 'FEATURE_ID', 'TRANSITION_ID', 'PRECURSOR_ID', 'PROTEIN_ID', 'GENE_ID', 'DECOY'])  # these columns should not be renamed
    renamed_columns = {c: [col if ' AS ' in col  else f"{c}.{col} AS '{c}.{col}'" if col not in whitelist else f"{c}.{col} AS '{col}'" for col in columns[c]] for c in columns.keys()}

    # create a list of all the columns
    columns_list = [col for c in renamed_columns.values() for col in c]

    # join the list into a single string separated by a comma and a space
    columnsToSelect = ", ".join(columns_list)


    ## Check if Gene tables are present
    if check_sqlite_table(con, "GENE"):
        gene_table_joins = '''
        LEFT JOIN PEPTIDE_GENE_MAPPING ON PEPTIDE.ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID
        LEFT JOIN GENE ON PEPTIDE_GENE_MAPPING.GENE_ID = GENE.ID
        '''
    else:
        gene_table_joins = ''
    

    # Check for Inferrence Context Scores
    if check_sqlite_table(con, "SCORE_PEPTIDE"):
        pepTable = getPeptideProteinScoreTable(condb, "peptide")
        pepJoin = 'LEFT JOIN pepTable ON pepTable.PEPTIDE_ID = PEPTIDE.ID'

    if check_sqlite_table(con, "SCORE_PROTEIN"):
        protTable = getPeptideProteinScoreTable(condb, "protein")
        protJoin = 'LEFT JOIN protTable ON protTable.PROTEIN_ID = PROTEIN.ID'

    # First read feature data
    # Feature Data
    if not transitionLevel:
        feature_query = f'''
        SELECT {columnsToSelect}
        FROM FEATURE
        FULL JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        LEFT JOIN RUN ON FEATURE.RUN_ID = RUN.ID
        LEFT JOIN FEATURE_MS1 ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
        LEFT JOIN FEATURE_MS2 ON FEATURE.ID = FEATURE_MS2.FEATURE_ID
        LEFT JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        LEFT JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        LEFT JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        LEFT JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
        {gene_table_joins}
        {pepJoin}
        {protJoin}
        '''
    else: # is transition level
        
        # merge transition and precursor level data
        feature_query = f'''
        SELECT {columnsToSelect}
        FROM FEATURE_TRANSITION
        FULL JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
        FULL JOIN FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
        FULL JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        LEFT JOIN RUN ON FEATURE.RUN_ID = RUN.ID
        LEFT JOIN FEATURE_MS1 ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
        LEFT JOIN FEATURE_MS2 ON FEATURE.ID = FEATURE_MS2.FEATURE_ID
        LEFT JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        LEFT JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        LEFT JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        LEFT JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
        {gene_table_joins}
        {pepJoin}
        {protJoin}
        '''
    condb.sql(feature_query).write_parquet(outfile)