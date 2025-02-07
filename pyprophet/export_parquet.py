import duckdb
import sqlite3
import pandas as pd
from pyprophet.export import check_sqlite_table
from duckdb_extensions import extension_importer
import re

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
        
    glob = conndb.sql(f"select {id}, SCORE, PVALUE, QVALUE, PEP from {score_table} where context == 'global'").df()
    glob.columns = [ col.upper() + '_GLOBAL' if col != id else col for col in glob.columns ]

    return glob, nonGlobal
    
def getVarColumnNames(condb, tableName):
    '''
    Get all the column names that start with VAR_ from the given table
    '''
    query = f"select name from pragma_table_info('{tableName}') where name like 'VAR_%'"
    return [ i[0] for i in condb.execute(query).fetchall() ]


# this method is only currently supported for combined output and not with ipf
def export_to_parquet(infile, outfile, transitionLevel=False, onlyFeatures=False, noDecoys=False):
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
    extension_importer.import_extension("sqlite_scanner")
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

    CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
    CREATE INDEX IF NOT EXISTS idx_transition_precursor_mapping_transition_id ON TRANSITION_PRECURSOR_MAPPING (TRANSITION_ID);
    CREATE INDEX IF NOT EXISTS idx_transition_precursor_mapping_precursor_id ON TRANSITION_PRECURSOR_MAPPING (PRECURSOR_ID);
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
    

  
    # check if IM columns exist
    query = con.execute("select count(*) as cntrec from pragma_table_info('feature_MS2') where name='EXP_IM'")
    hasIm = query.fetchone()[0] > 0
  
    # since do not want all of the columns (some columns are twice per table) manually select the columns want in a list, (note do not want decoy)
    # note TRAML_ID for precursor and transition are not the same
    columns = {}
    gene_table_joins = ''
    pepJoin = ''
    protJoin = ''
    ## library 
    columns['PRECURSOR'] = ['TRAML_ID', 'GROUP_LABEL', 'PRECURSOR_MZ', 'CHARGE', 'LIBRARY_INTENSITY', 'LIBRARY_RT', 'DECOY']
    if hasIm:
        columns['PRECURSOR'].append("LIBRARY_DRIFT_TIME")

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
    columns['FEATURE_MS2'] = ['FEATURE_ID', 'AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI'] + getVarColumnNames(con, 'FEATURE_MS2') 
    columns['FEATURE_MS1'] = ['APEX_INTENSITY', 'AREA_INTENSITY'] + getVarColumnNames(con, 'FEATURE_MS1')
    if hasIm:
        imColumns = ['EXP_IM', 'DELTA_IM']
        columns['FEATURE_MS2'] = columns['FEATURE_MS2'] + imColumns
        columns['FEATURE_MS1'] = columns['FEATURE_MS1'] + imColumns

    ### pyprophet scores 
    columns['SCORE_MS2'] = ["SCORE", "RANK", "PVALUE", "QVALUE", "PEP"]

    # Check for Peptide/Protein scores Context Scores
    if check_sqlite_table(con, "SCORE_PEPTIDE"):
        pepTable_global, pepTable_nonGlobal = getPeptideProteinScoreTable(condb, "peptide")
        pepJoin = '''LEFT JOIN pepTable_nonGlobal ON pepTable_nonGlobal.PEPTIDE_ID = PEPTIDE.ID and pepTable_nonGlobal.RUN_ID = RUN.ID
                     LEFT JOIN pepTable_global ON pepTable_global.PEPTIDE_ID = PEPTIDE.ID'''
        columns['pepTable_nonGlobal'] = list(set(pepTable_nonGlobal.columns).difference(set(['PEPTIDE_ID', 'RUN_ID']))) # all columns except PEPTIDE_ID and RUN_ID
        columns['pepTable_global'] = list(set(pepTable_global.columns).difference(set(['PEPTIDE_ID']))) # all columns except PEPTIDE_ID and RUN_ID


    if check_sqlite_table(con, "SCORE_PROTEIN"):
        protTable_global, protTable_nonGlobal = getPeptideProteinScoreTable(condb, "protein")
        protJoin = '''LEFT JOIN protTable_nonGlobal ON protTable_nonGlobal.PROTEIN_ID = PROTEIN.ID and protTable_nonGlobal.RUN_ID = RUN.ID
                      LEFT JOIN protTable_global ON protTable_global.PROTEIN_ID = PROTEIN.ID'''
        columns['protTable_nonGlobal'] = list(set(protTable_nonGlobal.columns).difference(set(['PROTEIN_ID', 'RUN_ID']))) # all columns except PROTEIN_ID and RUN_ID
        columns['protTable_global'] = list(set(protTable_global.columns).difference(set(['PROTEIN_ID']))) # all columns except PROTEIN_ID

    ## other
    columns['RUN'] = ['FILENAME']


    ## mappings
    columns['PRECURSOR_PEPTIDE_MAPPING'] = ['PEPTIDE_ID', 'PRECURSOR_ID']
    columns['PEPTIDE_PROTEIN_MAPPING'] = ['PROTEIN_ID']
    if check_sqlite_table(con, "GENE") and pd.read_sql('SELECT GENE_NAME FROM GENE', con).GENE_NAME[0]!='NA':
        columns['PEPTIDE_GENE_MAPPING'] = ['GENE_ID']

    # transition level
    if transitionLevel:
        columns['FEATURE_TRANSITION'] = ['AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI'] + getVarColumnNames(con, 'FEATURE_TRANSITION')
        columns['TRANSITION'] = ['TRAML_ID', 'PRODUCT_MZ', 'CHARGE', 'TYPE', 'ORDINAL', 'DETECTING', 'IDENTIFYING', 'QUANTIFYING', 'LIBRARY_INTENSITY']
        columns['TRANSITION_PRECURSOR_MAPPING'] = ['TRANSITION_ID']

    ### rename column names that are in common 
    whitelist = set(['PEPTIDE_ID', 'FEATURE_ID', 'TRANSITION_ID', 'PRECURSOR_ID', 'PROTEIN_ID', 'GENE_ID', 'DECOY', 'RUN_ID'])  # these columns should not be renamed
    
    for table in columns.keys(): # iterate through all tables
        ## rename pepTable and protTable to be inline with sql scheme
        if table in ['pepTable_nonGlobal','pepTable_global']:
            renamed_table = 'SCORE_PEPTIDE'
        elif table in ['protTable_nonGlobal', 'protTable_global']:
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
    
    # create a list of just aliases for groupby
    pattern = re.compile(r"(.*)\sAS")
    alias_list = [ pattern.search(col).group(1) for c in columns.values() for col in c]

    # join the list into a single string separated by a comma and a space
    columnsToSelect = ", ".join(columns_list)
    aliasToSelect = ", ".join(alias_list)

    # For feature level group important transition level data into one row separated by ';'
    featureLvlPrefix = "GROUP_CONCAT(TRANSITION.ID, ';') AS 'TRANSITION_ID', GROUP_CONCAT(TRANSITION.ANNOTATION, ';') AS 'TRANSITION_ANNOTATION'" if not transitionLevel else ""
    featureLvlSuffix = f'GROUP BY {aliasToSelect}' if not transitionLevel else ""

    decoyExclude = "WHERE PRECURSOR.DECOY == 0" if noDecoys else ""

    if not onlyFeatures:
        query = f'''
        SELECT {columnsToSelect},
        {featureLvlPrefix}
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
        {decoyExclude}
        {featureLvlSuffix}
        '''
    else:
        query = f'''
        SELECT {columnsToSelect},
        {featureLvlPrefix}
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
        {decoyExclude}
        {featureLvlSuffix}
        '''
    condb.sql(query).write_parquet(outfile)