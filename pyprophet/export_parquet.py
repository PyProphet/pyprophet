import duckdb
import sqlite3
import numpy as np
import pandas as pd
import polars as pl
from pyprophet.export import check_sqlite_table
from duckdb_extensions import extension_importer
import re

def load_sqlite_scanner():
    """
    Ensures the `sqlite_scanner` extension is installed and loaded in DuckDB.
    """
    try:
        duckdb.execute("LOAD sqlite_scanner")
    except Exception as e:
        if "Extension 'sqlite_scanner' not found" in str(e):
            try:
                duckdb.execute("INSTALL sqlite_scanner")
                duckdb.execute("LOAD sqlite_scanner")
            except Exception as install_error:
                if "already installed but the origin is different" in str(install_error):
                    duckdb.execute("FORCE INSTALL sqlite_scanner")
                    duckdb.execute("LOAD sqlite_scanner")
                else:
                    raise install_error
        else:
            raise e

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
    load_sqlite_scanner()
        
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


def convert_osw_to_parquet(infile, outfile, compression_method='zstd', compression_level=11):
    '''
    Convert an OSW sqlite file to Parquet format
    
    Note:
        This is different from the export_to_parquet function, this will create a parquet compatible with pyprophet scoring. The resulting parquet will have each row as a run-specific precursor feature, with transition level data collapsed into a single row as arrays. 
        
    Note:
        Some notes on performance (converting a non-scored OSW file with 4,130 precursors, 114,295 features, 316,825 transitions and 6 runs):
        
        merged.osw (764M)
        merged_osw_snappy.parquet (355M)
        merged_osw_gzip.parquet (284M)
        merged_osw_zstd.parquet (301M)
        merged_osw_zstd_level_11_33s.parquet (289M, exported in 33.2829 seconds, 62% decrease in size)
        merged_osw_zstd_level_22_10m_51s.parquet (277M)
        merged_osw_brotli.parquet (275M)
        merged_osw_brotli_level_11_27m_45s.parquet (255M)
        
    See: [polars.DataFrame.write_parquet](https://docs.pola.rs/api/python/stable/reference/api/polars.DataFrame.write_parquet.html) for more information on compression methods and levels.
        

    Parameters:
        infile: (str) path to osw sqlite file
        outfile: (str) path to write out parquet file
        compression_method: (str) compression method for parquet file (default: 'zstd')
        compression_level: (int) compression level for parquet file (default: 11)
    
    Return:
        None
    '''

    load_sqlite_scanner()
    conn = duckdb.connect(database=infile, read_only=True)

    # Get Gene/Protein/Peptide/Precursor table
    query = """
    SELECT 
        PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID AS PROTEIN_ID,
        PEPTIDE.ID AS PEPTIDE_ID,
        PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID AS PRECURSOR_ID,
        PROTEIN.PROTEIN_ACCESSION AS PROTEIN_ACCESSION,
        PEPTIDE.UNMODIFIED_SEQUENCE,
        PEPTIDE.MODIFIED_SEQUENCE,
        PRECURSOR.TRAML_ID AS PRECURSOR_TRAML_ID,
        PRECURSOR.GROUP_LABEL AS PRECURSOR_GROUP_LABEL,
        PRECURSOR.PRECURSOR_MZ AS PRECURSOR_MZ,
        PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
        PRECURSOR.LIBRARY_INTENSITY AS PRECURSOR_LIBRARY_INTENSITY,
        PRECURSOR.LIBRARY_RT AS PRECURSOR_LIBRARY_RT
        {library_drift_time_field}
        {gene_fields},
        PROTEIN.DECOY AS PROTEIN_DECOY,
        PEPTIDE.DECOY AS PEPTIDE_DECOY,
        PRECURSOR.DECOY AS PRECURSOR_DECOY
    FROM PRECURSOR
    INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
    INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
    INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
    INNER JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
    {gene_joins}
    """

    # # Conditionally include fields and joins (Older OSW files may not have GENE table, and may not have ion mobility related columns)
    gene_tables_exist = all(
        table
        in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        .df()["name"]
        .values
        for table in ["PEPTIDE_GENE_MAPPING", "GENE"]
    )
    precursor_columns = conn.execute("PRAGMA table_info(PRECURSOR)").df()['name'].values
    has_library_drift_time = 'LIBRARY_DRIFT_TIME' in precursor_columns

    query = query.format(
        library_drift_time_field=""",
        PRECURSOR.LIBRARY_DRIFT_TIME AS PRECURSOR_LIBRARY_DRIFT_TIME""" if has_library_drift_time else """,
        NULL AS PRECURSOR_LIBRARY_DRIFT_TIME""",
        
        gene_fields=""",
        PEPTIDE_GENE_MAPPING.GENE_ID AS GENE_ID,
        GENE.GENE_NAME AS GENE_NAME,
        GENE.DECOY AS GENE_DECOY""" if gene_tables_exist else """,
        NULL AS GENE_ID,
        NULL AS GENE_NAME,
        NULL AS GENE_DECOY""",
        
        gene_joins="""
        LEFT JOIN PEPTIDE_GENE_MAPPING ON PEPTIDE.ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID
        LEFT JOIN GENE ON PEPTIDE_GENE_MAPPING.GENE_ID = GENE.ID""" if gene_tables_exist else ""
    )
    precursor_df = conn.execute(query).pl()

    # Get Transition table
    transition_columns = conn.execute("PRAGMA table_info(TRANSITION)").df()['name'].values
    has_annotation = 'ANNOTATION' in transition_columns
    query = f"""
    SELECT 
        TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID AS PRECURSOR_ID,
        TRANSITION.ID AS TRANSITION_ID,
        TRANSITION.TRAML_ID AS TRANSITION_TRAML_ID,
        TRANSITION.PRODUCT_MZ,
        TRANSITION.CHARGE AS TRANSITION_CHARGE,
        TRANSITION.TYPE AS TRANSITION_TYPE,
        TRANSITION.ORDINAL AS TRANSITION_ORDINAL,
        {'TRANSITION.ANNOTATION AS TRANSITION_ANNOTATION,' if has_annotation else 'NULL AS TRANSITION_ANNOTATION,'}
        TRANSITION.DETECTING AS TRANSITION_DETECTING,
        TRANSITION.LIBRARY_INTENSITY AS TRANSITION_LIBRARY_INTENSITY,
        TRANSITION.DECOY AS TRANSITION_DECOY
    FROM TRANSITION
    INNER JOIN TRANSITION_PRECURSOR_MAPPING ON TRANSITION.ID = TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID
    """
    transition_df = conn.execute(query).pl()
    if not has_annotation:
        transition_df = transition_df.with_columns(
            pl.concat_str([
                pl.col("TRANSITION_TYPE"),
                pl.col("TRANSITION_ORDINAL").cast(pl.Utf8),
                pl.lit("^"),
                pl.col("TRANSITION_CHARGE").cast(pl.Utf8)
            ], separator="").alias("ANNOTATION")
        )

        transition_df = transition_df.select([
            "PRECURSOR_ID",
            "TRANSITION_ID",
            "TRANSITION_TRAML_ID",
            "PRODUCT_MZ",
            "TRANSITION_CHARGE",
            "TRANSITION_TYPE",
            "TRANSITION_ORDINAL",
            "ANNOTATION",
            "TRANSITION_DETECTING",
            "TRANSITION_LIBRARY_INTENSITY",
            "TRANSITION_DECOY"
        ])

    # Get Transition-Peptide table
    query = """
    SELECT 
        TRANSITION_ID,
        PEPTIDE_ID AS PEPTIDE_IPF_ID
    FROM TRANSITION_PEPTIDE_MAPPING
    """
    transition_peptide_df = conn.execute(query).pl()

    # Merge transition_df and transition_peptide_df on TRANSITION_ID
    transition_df = transition_df.join(transition_peptide_df, on="TRANSITION_ID", how="full", coalesce=True)

    # Get Feature table
    feature_columns = conn.execute("PRAGMA table_info(FEATURE)").df()['name'].values
    has_im = 'EXP_IM' in feature_columns
    query = f"""
    SELECT
    FEATURE.RUN_ID AS RUN_ID,
    RUN.FILENAME,
    FEATURE.PRECURSOR_ID AS PRECURSOR_ID,
    FEATURE.ID AS FEATURE_ID,
    FEATURE.EXP_RT,
    {'FEATURE.EXP_IM,' if has_im else 'NULL AS EXP_IM,'}
    FEATURE.NORM_RT,
    FEATURE.DELTA_RT,
    FEATURE.LEFT_WIDTH,
    FEATURE.RIGHT_WIDTH
    FROM FEATURE
    INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID
    """
    feature_df = conn.execute(query).pl()
    feature_df = feature_df[[s.name for s in feature_df if not (s.null_count() == feature_df.height)]]

    # Get FEATURE_MS1
    query = """
    SELECT
    *
    FROM FEATURE_MS1
    """
    feature_ms1_df = conn.execute(query).pl()
    feature_ms1_df = feature_ms1_df[[s.name for s in feature_ms1_df if not (s.null_count() == feature_ms1_df.height)]]
    # Append "FEATURE_MS1_" to column names
    feature_ms1_df = feature_ms1_df.rename({col: f"FEATURE_MS1_{col}" for col in feature_ms1_df.columns if col != "FEATURE_ID"})

    # Get FEATURE_MS2
    query = """
    SELECT
    *
    FROM FEATURE_MS2
    """
    feature_ms2_df = conn.execute(query).pl()
    feature_ms2_df = feature_ms2_df[[s.name for s in feature_ms2_df if not (s.null_count() == feature_ms2_df.height)]]
    # Append "FEATURE_MS2_" to column names
    feature_ms2_df = feature_ms2_df.rename({col: f"FEATURE_MS2_{col}" for col in feature_ms2_df.columns if col != "FEATURE_ID"})

    # Get FEATURE_TRANSITION
    query = """
    SELECT
    * FROM FEATURE_TRANSITION
    """
    feature_transition_df = conn.execute(query).pl()
    feature_transition_df = feature_transition_df[[s.name for s in feature_transition_df if not (s.null_count() == feature_transition_df.height)]]
    # Append "FEATURE_TRANSITION_" to column names
    new_columns = {
        col: f"FEATURE_TRANSITION_{col}" 
        for col in feature_transition_df.columns 
        if col not in ["FEATURE_ID", "TRANSITION_ID"]
    }
    feature_transition_df = feature_transition_df.rename(new_columns)
    feature_transition_df = feature_transition_df.with_columns(pl.col("FEATURE_ID").cast(pl.Utf8))

    ## Merge feature_transition_df with transition_df ON TRANSITION_ID
    feature_transition_df = transition_df.join(feature_transition_df, on="TRANSITION_ID", how="left")

    ## Patch missing FEATURE_ID in feature_transition_df where some transitions are not scored
    feature_transition_not_scored = feature_transition_df.filter(
        pl.col("FEATURE_ID").is_null()
    ).select([
        "PRECURSOR_ID", "TRANSITION_ID", "TRANSITION_TRAML_ID", "PRODUCT_MZ",
        "TRANSITION_CHARGE", "TRANSITION_TYPE", "TRANSITION_ORDINAL",
        "ANNOTATION", "TRANSITION_DETECTING", "TRANSITION_LIBRARY_INTENSITY",
        "TRANSITION_DECOY"
    ])

    # Get Unique Precursor_ID, FEATURE_ID
    precursor_feature_df = feature_transition_df.select(["PRECURSOR_ID", "FEATURE_ID"]).unique()

    # merge precursor_feature_df with feature_transition_not_scored to propagate the missing FEATURE_ID
    feature_transition_not_scored_patched = precursor_feature_df.join(
        feature_transition_not_scored, 
        on="PRECURSOR_ID", 
        how="left"
    ).filter(
        pl.col("TRANSITION_ID").is_not_null()
    ).with_columns(
        pl.col("TRANSITION_ID").cast(pl.Int64)
    )

    # Remove rows where FEATURE_ID is NaN from feature_transition_df, and then merge with feature_transition_not_scored_patched
    feature_transition_df = feature_transition_df.filter(
        pl.col("FEATURE_ID").is_not_null()
    ).join(
        feature_transition_not_scored_patched,
        on=["PRECURSOR_ID", "FEATURE_ID", "TRANSITION_ID", "TRANSITION_TRAML_ID",
            "PRODUCT_MZ", "TRANSITION_CHARGE", "TRANSITION_TYPE",
            "TRANSITION_ORDINAL", "ANNOTATION", "TRANSITION_DETECTING",
            "TRANSITION_LIBRARY_INTENSITY", "TRANSITION_DECOY"],
        how="full",
        coalesce=True
    )

    # Collapse by grouping by PRECURSOR_ID and FEATURE_ID
    feature_transition_df = feature_transition_df.group_by(["PRECURSOR_ID", "FEATURE_ID"]).agg(pl.all())

    # Merge feature_df, feature_ms1_df, feature_ms2_df ON FEATURE_ID
    feature_df = feature_df.join(feature_ms1_df, on="FEATURE_ID", how="left", coalesce=True)
    feature_df = feature_df.join(feature_ms2_df, on="FEATURE_ID", how="left", coalesce=True)
    feature_df = feature_df.with_columns(
        pl.col("FEATURE_ID").cast(pl.Utf8),
        pl.col("RUN_ID").cast(pl.Utf8)
    )

    # Merge collapsed transition feature data
    feature_df = feature_df.join(
        feature_transition_df,
        on=["PRECURSOR_ID", "FEATURE_ID"],
        how="left", 
        coalesce=True
    )

    # Merge precursor data with feature data
    master_df = precursor_df.join(
        feature_df, 
        on="PRECURSOR_ID", 
        how="full",
        coalesce=True
    )
    master_df = master_df.filter(
        ~(pl.col("RUN_ID").is_null() & pl.col("FEATURE_ID").is_null())
    )
    master_df = master_df[[s.name for s in master_df if not (s.null_count() == master_df.height)]]
    master_df = master_df.with_columns(
        pl.col("RUN_ID").cast(pl.Int64),
        pl.col("FEATURE_ID").cast(pl.Int64)
    )

    conn.close()

    # Write to parquet
    master_df.write_parquet(
        outfile,
        compression=compression_method,
        compression_level=compression_level
    )


def convert_sqmass_to_parquet(
    infile, outfile, oswfile, compression_method="zstd", compression_level=11
):
    '''
    Convert a SQMass sqlite file to Parquet format
    '''
    load_sqlite_scanner()
    xic_conn = duckdb.connect(database=infile, read_only=True)
    osw_conn = duckdb.connect(database=oswfile, read_only=True)

    query = """
    SELECT
    PRECURSOR.ID AS PRECURSOR_ID,
    TRANSITION.ID AS TRANSITION_ID,
    PEPTIDE.MODIFIED_SEQUENCE,
    PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
    TRANSITION.CHARGE AS PRODUCT_CHARGE,
    TRANSITION.DETECTING AS DETECTING_TRANSITION,
    PRECURSOR.DECOY AS PRECURSOR_DECOY,
    TRANSITION.DECOY AS PRODUCT_DECOY,
    FROM PRECURSOR
    INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
    INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
    INNER JOIN TRANSITION_PRECURSOR_MAPPING ON PRECURSOR.ID = TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID
    INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
    """
    osw_df = osw_conn.execute(query).pl()

    query = """
    SELECT
    CHROMATOGRAM.NATIVE_ID,
    DATA.COMPRESSION,
    DATA.DATA_TYPE,
    DATA.DATA
    FROM CHROMATOGRAM
    INNER JOIN DATA ON DATA.CHROMATOGRAM_ID = CHROMATOGRAM.ID
    """
    chrom_df = xic_conn.execute(query).pl()
    
    xic_conn.close()
    osw_conn.close()

    chrom_df_prec = (
        chrom_df.filter(chrom_df["NATIVE_ID"].str.contains("_Precursor_i\\d+"))
        # Add column with just the precursor id - now operating on the filtered DataFrame
        .with_columns(
            pl.col("NATIVE_ID")
            .str.extract(r"(\d+)_Precursor_i\d+")
            .alias("PRECURSOR_ID")
            .cast(pl.Int64)
        )
    )

    chrom_df_trans = (
        chrom_df.filter(~chrom_df["NATIVE_ID"].str.contains("_Precursor_i\\d+"))
        # Add column with just the transition id - now operating on the filtered DataFrame
        .with_columns(pl.col("NATIVE_ID").alias("TRANSITION_ID").cast(pl.Int64))
    )

    # Join osw_df and chrom_df_prec on PRECURSOR_ID
    chrom_df_prec_m = (
        osw_df.select(
            ["PRECURSOR_ID", "MODIFIED_SEQUENCE", "PRECURSOR_CHARGE", "PRECURSOR_DECOY"]
        )
        .unique()
        .with_columns(
            pl.lit(None).cast(pl.Int64).alias("TRANSITION_ID"),
            pl.lit(None).cast(pl.Int64).alias("PRODUCT_CHARGE"),
            pl.lit(1).cast(pl.Int64).alias("DETECTING_TRANSITION"),
            pl.lit(None).cast(pl.Int64).alias("PRODUCT_DECOY"),
        )
        .select(
            [
                "PRECURSOR_ID",
                "TRANSITION_ID",
                "MODIFIED_SEQUENCE",
                "PRECURSOR_CHARGE",
                "PRODUCT_CHARGE",
                "DETECTING_TRANSITION",
                "PRECURSOR_DECOY",
                "PRODUCT_DECOY",
            ]
        )
        .join(
            chrom_df_prec,
            left_on="PRECURSOR_ID",
            right_on="PRECURSOR_ID",
            how="inner",
            coalesce=True,
        )
    )

    chrom_df_trans_m = osw_df.join(
        chrom_df_trans,
        left_on="TRANSITION_ID",
        right_on="TRANSITION_ID",
        how="inner",
        coalesce=True,
    )

    chrom_df_m = chrom_df_prec_m.vstack(chrom_df_trans_m).sort(by=["PRECURSOR_ID"])

    chrom_df_restructured = chrom_df_m.pivot(
        values=["DATA", "COMPRESSION"],
        index=[
            "PRECURSOR_ID",
            "TRANSITION_ID",
            "MODIFIED_SEQUENCE",
            "PRECURSOR_CHARGE",
            "PRODUCT_CHARGE",
            "DETECTING_TRANSITION",
            "PRECURSOR_DECOY",
            "PRODUCT_DECOY",
            "NATIVE_ID",
        ],
        on="DATA_TYPE",
        aggregate_function="first",
    )

    chrom_df_restructured = chrom_df_restructured.rename(
        {
            "DATA_1": "INTENSITY_DATA",
            "DATA_2": "RT_DATA",
            "COMPRESSION_1": "INTENSITY_COMPRESSION",
            "COMPRESSION_2": "RT_COMPRESSION",
        }
    )
    
    chrom_df_restructured.write_parquet(
        outfile,
        compression=compression_method,
        compression_level=compression_level,
    )
