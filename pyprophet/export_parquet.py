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
    nonGlobal.columns = [ col.upper().replace('-', '') for col in nonGlobal.columns.map('_'.join)]
    nonGlobal= nonGlobal.reset_index()
        
    glob = conndb.sql(f"select {id}, RUN_ID, SCORE, PVALUE, QVALUE, PEP from {score_table} where context == 'global'").df()
    glob.columns = [ col.upper() + '_GLOBAL' if col != id else col for col in glob.columns ]

    return nonGlobal.merge(glob, how='outer')
    
def getVarColumnNames(condb, tableName):
    '''
    Get all the column names that start with VAR_ from the given table
    '''
    query = f"select name from pragma_table_info('{tableName}') where name like 'VAR_%'"
    return [ i[0] for i in condb.execute(query).fetchall() ]


# this method is only currently supported for combined output and not with ipf
def export_to_parquet(infile, outfile, transitionLevel, onlyFeatures=False):
    '''
    Convert an OSW sqlite file to Parquet format

    Parameters:
        infile: (str) path to osw sqlite file
        outfile: (str) path to write out parquet file
        transitionLevel: (bool) append transition level data
        onlyFeatures: (bool) Only output precursors associated with a feature
    
    Return:
        None
    '''
    condb = duckdb.connect(infile)
    con = sqlite3.connect(infile)

    print(getVarColumnNames(condb, 'FEATURE_MS2'))

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

 
    ## Check if Gene tables are present
    if check_sqlite_table(con, "GENE"):
        gene_table_joins = '''
        LEFT JOIN PEPTIDE_GENE_MAPPING ON PEPTIDE.ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID
        LEFT JOIN GENE ON PEPTIDE_GENE_MAPPING.GENE_ID = GENE.ID
        '''
    else:
        gene_table_joins = ''
    

  
   
    # since do not want all of the columns (some columns are twice per table) manually select the columns want in a list, (note do not want decoy)
    # note TRAML_ID for precursor and transition are not the same
    columns = {}
    gene_table_joins = ''
    pepJoin = ''
    protJoin = ''
    ## library 
    columns['PRECURSOR'] = ['TRAML_ID', 'GROUP_LABEL', 'PRECURSOR_MZ', 'CHARGE', 'LIBRARY_INTENSITY', 'LIBRARY_RT', 'LIBRARY_DRIFT_TIME', 'DECOY']
    columns['PEPTIDE'] = ['UNMODIFIED_SEQUENCE', 'MODIFIED_SEQUENCE']
    columns['PROTEIN'] = ['PROTEIN_ACCESSION']

    if check_sqlite_table(con, "GENE") and pd.read_sql('SELECT GENE_NAME FROM GENE', con).GENE_NAME[0]!='NA':
        columns['GENE'] = ['GENE_NAME']
        # add gene table joins
        gene_table_joins = '''
        LEFT JOIN PEPTIDE_GENE_MAPPING ON PEPTIDE.ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID
        LEFT JOIN GENE ON PEPTIDE_GENE_MAPPING.GENE_ID = GENE.ID
        '''

    ## features
    columns['FEATURE'] = ['EXP_RT', 'EXP_IM', 'NORM_RT', 'DELTA_RT', 'LEFT_WIDTH', 'RIGHT_WIDTH']
    columns['FEATURE_MS2'] = ['FEATURE_ID', 'AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI'] + getVarColumnNames(condb, 'FEATURE_MS2') 
    columns['FEATURE_MS1'] = ['APEX_INTENSITY', 'VAR_MASSDEV_SCORE'] + getVarColumnNames(condb, 'FEATURE_MS1')

    # check if IM columns exist
    query = con.execute("select count(*) as cntrec from pragma_table_info('feature_MS2') where name='EXP_IM'")
    hasIm = query.fetchone()[0] > 0
    if hasIm:
        imColumns = ['EXP_IM', 'DELTA_IM']
        columns['FEATURE_MS2'] = columns['FEATURE_MS2'] + imColumns
        columns['FEATURE_MS1'] = columns['FEATURE_MS1'] + imColumns

    ### pyprophet scores 
    columns['SCORE_MS2'] = ["SCORE", "RANK", "PVALUE", "QVALUE", "PEP"]

    # Check for Peptide/Protein scores Context Scores
    if check_sqlite_table(con, "SCORE_PEPTIDE"):
        pepTable = getPeptideProteinScoreTable(condb, "peptide")
        pepJoin = 'LEFT JOIN pepTable ON pepTable.PEPTIDE_ID = PEPTIDE.ID and pepTable.RUN_ID = RUN.ID'
        columns['pepTable'] = list(set(pepTable.columns).difference(set(['PEPTIDE_ID', 'RUN_ID']))) # all columns except PEPTIDE_ID and RUN_ID


    if check_sqlite_table(con, "SCORE_PROTEIN"):
        protTable = getPeptideProteinScoreTable(condb, "protein")
        protJoin = 'LEFT JOIN protTable ON protTable.PROTEIN_ID = PROTEIN.ID and protTable.RUN_ID = RUN.ID'
        columns['protTable'] = list(set(protTable.columns).difference(set(['PROTEIN_ID', 'RUN_ID']))) # all columns except PEPTIDE_ID and RUN_ID

    ## other
    columns['RUN'] = ['FILENAME']


    ## mappings
    columns['PRECURSOR_PEPTIDE_MAPPING'] = ['PEPTIDE_ID', 'PRECURSOR_ID']
    columns['PEPTIDE_PROTEIN_MAPPING'] = ['PROTEIN_ID']
    if check_sqlite_table(con, "GENE") and pd.read_sql('SELECT GENE_NAME FROM GENE', con).GENE_NAME[0]!='NA':
        columns['PEPTIDE_GENE_MAPPING'] = ['GENE_ID']

    # transition level
    if transitionLevel:
        columns['FEATURE_TRANSITION'] = ['AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI'] + getVarColumnNames(condb, 'FEATURE_TRANSITION')
        columns['TRANSITION'] = ['TRAML_ID', 'PRODUCT_MZ', 'CHARGE', 'TYPE', 'ORDINAL', 'DETECTING', 'IDENTIFYING', 'QUANTIFYING', 'LIBRARY_INTENSITY']
        columns['TRANSITION_PRECURSOR_MAPPING'] = ['TRANSITION_ID']

    ### rename column names that are in common 
    whitelist = set(['PEPTIDE_ID', 'FEATURE_ID', 'TRANSITION_ID', 'PRECURSOR_ID', 'PROTEIN_ID', 'GENE_ID', 'DECOY', 'RUN_ID'])  # these columns should not be renamed
    
    for table in columns.keys(): # iterate through all tables
        ## rename pepTable and protTable to be inline with sql scheme
        if table == 'pepTable':
            renamed_table = 'SCORE_PEPTIDE'
        elif table == 'protTable':
            renamed_table = 'SCORE_PROTEIN'
        else:
            renamed_table = table
        for c_idx, c in enumerate(columns[table]): # iterate through all columns in the table
            if c in whitelist:
                columns[table][c_idx] = f"{table}.{c} AS {c}"
            else:
                columns[table][c_idx] = f"{table}.{c} AS '{renamed_table}.{c}'"


    # create a list of all the columns
    columns_list = [col for c in columns.values() for col in c]

    # join the list into a single string separated by a comma and a space
    columnsToSelect = ", ".join(columns_list)

    join_features = "LEFT JOIN" if onlyFeatures else "FULL JOIN"

    # First read feature data
    # Feature Data
    if not transitionLevel:
        feature_query = f'''
        SELECT {columnsToSelect}
        FROM FEATURE
        {join_features} PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
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
        if not onlyFeatures:
            feature_query = f'''
            SELECT {columnsToSelect}
            FROM TRANSITION_PRECURSOR_MAPPING
            LEFT JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
            LEFT JOIN PRECURSOR ON TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID = PRECURSOR.ID
            LEFT JOIN FEATURE_TRANSITION ON TRANSITION.ID = FEATURE_TRANSITION.TRANSITION_ID 
            LEFT JOIN FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
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
        else:
            feature_query = f'''
            SELECT {columnsToSelect}
            FROM FEATURE_TRANSITION 
            LEFT JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
            LEFT JOIN FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
            LEFT JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            LEFT JOIN TRANSITION_PRECURSOR_MAPPING ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
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