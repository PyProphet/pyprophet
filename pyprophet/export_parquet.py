import gc
import os
import time
import duckdb
import sqlite3
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from pyprophet.export import check_sqlite_table
from duckdb_extensions import extension_importer
import re
import click
from pyprophet.data_handling import format_bytes

def load_sqlite_scanner(conn: duckdb.DuckDBPyConnection):
    """
    Ensures the `sqlite_scanner` extension is installed and loaded in DuckDB.
    """
    try:
        conn.execute("LOAD sqlite_scanner")
    except Exception as e:
        if "Extension 'sqlite_scanner' not found" in str(e):
            try:
                conn.execute("INSTALL sqlite_scanner")
                conn.execute("LOAD sqlite_scanner")
            except Exception as install_error:
                if "already installed but the origin is different" in str(install_error):
                    conn.execute("FORCE INSTALL sqlite_scanner")
                    conn.execute("LOAD sqlite_scanner")
                else:
                    raise install_error
        else:
            raise e

def get_table_columns(sqlite_file: str, table: str) -> list:
    with sqlite3.connect(sqlite_file) as conn:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]

def write_parquet_batches(
    df: pl.DataFrame, path: str, 
    row_group_size: int = 100_000,
    compression_method: str = "zstd",
    compression_level: int = 11,
):
    print(f"Info: number of rows: {df.height} and number of columns: {df.width}")
    table = df.to_arrow()
    writer = pq.ParquetWriter(
        path,
        table.schema,
        compression=compression_method,
        use_dictionary=True,
        compression_level=compression_level,
    )
    total_rows = df.height
    process = psutil.Process(os.getpid())
    for start in range(0, total_rows, row_group_size):
        end = min(start + row_group_size, total_rows)
        batch = table.slice(start, end - start)
        mem_before_batch = process.memory_info().rss
        writer.write_table(batch)
        gc.collect()
        mem_after_batch = process.memory_info().rss
        click.echo(
            f"  - [{ round(start/row_group_size)+1 } / {round(total_rows/row_group_size)+1}] Wrote rows {start}–{end}, memory delta: { format_bytes(max(0, (mem_after_batch - mem_before_batch))) }"
        )
    writer.close()

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
    
    condb = duckdb.connect(infile)
    load_sqlite_scanner(condb)
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


def convert_osw_to_parquet(
    infile: str,
    outfile: str,
    compression_method: str = "zstd",
    compression_level: int = 11,
    split_transition_data: bool = True
):
    """
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

    Note:
        If `split_transition_data` is set to True, the output will be a directory with two parquet files: `precursors_features.parquet` and `transition_features.parquet`. If set to False, the output will be a single parquet file with all data combined.

    Parameters:
        infile: (str) path to osw sqlite file
        outfile: (str) path to write out parquet file
        compression_method: (str) compression method for parquet file (default: 'zstd')
        compression_level: (int) compression level for parquet file (default: 11)
        split_transition_data: (bool) if True, will split the transition data into a separate file. If False, will combine the transition data with the precursor data. (default: True)

    Return:
        None
    """

    conn = duckdb.connect(":memory:")
    load_sqlite_scanner(conn)

    with sqlite3.connect(infile) as sql_conn:
        table_names = set(
            row[0]
            for row in sql_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )
        gene_tables_exist = {"PEPTIDE_GENE_MAPPING", "GENE"}.issubset(table_names)
        precursor_columns = get_table_columns(infile, "PRECURSOR")
        transition_columns = get_table_columns(infile, "TRANSITION")
        feature_columns = get_table_columns(infile, "FEATURE")

    has_library_drift_time = "LIBRARY_DRIFT_TIME" in precursor_columns
    has_annotation = "ANNOTATION" in transition_columns
    has_im = "EXP_IM" in feature_columns
    
    # Prepare feature ms1 columns for sql query
    feature_ms1_cols = get_table_columns(infile, "FEATURE_MS1")
    feature_ms1_cols.remove("FEATURE_ID")
    feature_ms1_cols = [f"{col} AS FEATURE_MS1_{col}" for col in feature_ms1_cols]
    feature_ms1_cols_sql = ', '.join([f"FEATURE_MS1.{col}" for col in feature_ms1_cols])

    # Prepare feature ms2 columns for sql query
    feature_ms2_cols = get_table_columns(infile, "FEATURE_MS2")
    feature_ms2_cols.remove("FEATURE_ID")
    feature_ms2_cols = [f"{col} AS FEATURE_MS2_{col}" for col in feature_ms2_cols]
    feature_ms2_cols_sql = ', '.join([f"FEATURE_MS2.{col}" for col in feature_ms2_cols])
    
    # Prepare feature transition columns for sql query
    feature_transition_cols = get_table_columns(infile, "FEATURE_TRANSITION")
    feature_transition_cols.remove("FEATURE_ID")
    feature_transition_cols.remove("TRANSITION_ID")
    feature_transition_cols = ["FEATURE_ID"] + [f"{col} AS FEATURE_TRANSITION_{col}" for col in feature_transition_cols]
    feature_transition_cols_sql = ', '.join([f"FEATURE_TRANSITION.{col}" for col in feature_transition_cols])
        
    if split_transition_data:
        os.makedirs(outfile, exist_ok=True)
        
        click.echo("Info: Writing precursor data...")
        
        path = os.path.join(outfile, "precursors_features.parquet")
        
        precursor_template = """
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
            PRECURSOR.LIBRARY_RT AS PRECURSOR_LIBRARY_RT,
            {library_drift_time} AS PRECURSOR_LIBRARY_DRIFT_TIME,
            {gene_id} AS GENE_ID,
            {gene_name} AS GENE_NAME,
            {gene_decoy} AS GENE_DECOY,
            PROTEIN.DECOY AS PROTEIN_DECOY,
            PEPTIDE.DECOY AS PEPTIDE_DECOY,
            PRECURSOR.DECOY AS PRECURSOR_DECOY,
            FEATURE.RUN_ID AS RUN_ID,
            RUN.FILENAME,
            FEATURE.ID AS FEATURE_ID,
            FEATURE.EXP_RT,
            {exp_im} AS EXP_IM,
            FEATURE.NORM_RT,
            FEATURE.DELTA_RT,
            FEATURE.LEFT_WIDTH,
            FEATURE.RIGHT_WIDTH,
            {feature_ms1_cols_sql},
            {feature_ms2_cols_sql}
        FROM sqlite_scan('{infile}', 'PRECURSOR') AS PRECURSOR
        INNER JOIN sqlite_scan('{infile}', 'PRECURSOR_PEPTIDE_MAPPING') AS PRECURSOR_PEPTIDE_MAPPING 
            ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        INNER JOIN sqlite_scan('{infile}', 'PEPTIDE') AS PEPTIDE 
            ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN sqlite_scan('{infile}', 'PEPTIDE_PROTEIN_MAPPING') AS PEPTIDE_PROTEIN_MAPPING 
            ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        INNER JOIN sqlite_scan('{infile}', 'PROTEIN') AS PROTEIN 
            ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
        {gene_joins}
        INNER JOIN sqlite_scan('{infile}', 'FEATURE') AS FEATURE 
            ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN sqlite_scan('{infile}', 'FEATURE_MS1') AS FEATURE_MS1 
            ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
        INNER JOIN sqlite_scan('{infile}', 'FEATURE_MS2') AS FEATURE_MS2 
            ON FEATURE.ID = FEATURE_MS2.FEATURE_ID
        INNER JOIN sqlite_scan('{infile}', 'RUN') AS RUN 
            ON FEATURE.RUN_ID = RUN.ID
        """

        precursor_query = precursor_template.format(
            infile=infile,
            library_drift_time="PRECURSOR.LIBRARY_DRIFT_TIME" if has_library_drift_time else "NULL",
            gene_id="PEPTIDE_GENE_MAPPING.GENE_ID" if gene_tables_exist else "NULL",
            gene_name="GENE.GENE_NAME" if gene_tables_exist else "NULL",
            gene_decoy="GENE.DECOY" if gene_tables_exist else "NULL",
            gene_joins=(
                "LEFT JOIN sqlite_scan('{infile}', 'PEPTIDE_GENE_MAPPING') AS PEPTIDE_GENE_MAPPING "
                "ON PEPTIDE.ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID "
                "LEFT JOIN sqlite_scan('{infile}', 'GENE') AS GENE "
                "ON PEPTIDE_GENE_MAPPING.GENE_ID = GENE.ID"
            ).format(infile=infile) if gene_tables_exist else "",
            exp_im="FEATURE.EXP_IM" if has_im else "NULL",
            feature_ms1_cols_sql=feature_ms1_cols_sql,
            feature_ms2_cols_sql=feature_ms2_cols_sql,
        )

        copy_query = f"COPY ({precursor_query}) TO '{path}' (FORMAT 'parquet', COMPRESSION '{compression_method}', COMPRESSION_LEVEL {compression_level});"

        # Execute the COPY query in DuckDB — this streams directly to disk
        conn.execute(copy_query)

        click.echo(
            "Info: Writing transition data..."
        )
        
        path = os.path.join(outfile, "transition_features.parquet")

        transition_template = """
        SELECT 
            TRANSITION_PEPTIDE_MAPPING.PEPTIDE_ID AS IPF_PEPTIDE_ID,
            TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID AS PRECURSOR_ID,
            TRANSITION.ID AS TRANSITION_ID,
            TRANSITION.TRAML_ID AS TRANSITION_TRAML_ID,
            TRANSITION.PRODUCT_MZ,
            TRANSITION.CHARGE AS TRANSITION_CHARGE,
            TRANSITION.TYPE AS TRANSITION_TYPE,
            TRANSITION.ORDINAL AS TRANSITION_ORDINAL,
            {annotation} AS ANNOTATION,
            TRANSITION.DETECTING AS TRANSITION_DETECTING,
            TRANSITION.LIBRARY_INTENSITY AS TRANSITION_LIBRARY_INTENSITY,
            TRANSITION.DECOY AS TRANSITION_DECOY,
            {feature_transition_cols_sql}
        FROM sqlite_scan('{infile}', 'TRANSITION') AS TRANSITION
        INNER JOIN sqlite_scan('{infile}', 'TRANSITION_PRECURSOR_MAPPING') AS TRANSITION_PRECURSOR_MAPPING 
            ON TRANSITION.ID = TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID
        FULL JOIN sqlite_scan('{infile}', 'TRANSITION_PEPTIDE_MAPPING') AS TRANSITION_PEPTIDE_MAPPING 
            ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
        FULL JOIN sqlite_scan('{infile}', 'FEATURE_TRANSITION') AS FEATURE_TRANSITION 
            ON TRANSITION.ID = FEATURE_TRANSITION.TRANSITION_ID
        """

        transition_query = transition_template.format(
            infile=infile,
            annotation=(
                "TRANSITION.ANNOTATION"
                if has_annotation
                else "TRANSITION.TYPE || CAST(TRANSITION.ORDINAL AS VARCHAR) || '^' || CAST(TRANSITION.CHARGE AS VARCHAR)"
            ),
            feature_transition_cols_sql=feature_transition_cols_sql,
        )

        copy_transition_query = f"COPY ({transition_query}) TO '{path}' (FORMAT 'parquet', COMPRESSION '{compression_method}', COMPRESSION_LEVEL {compression_level});"

        conn.execute(copy_transition_query)


    else:
        click.echo(
            "Info: Writing both precursor and transition data to a single parquet file..."
        )

        template = """
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
            PRECURSOR.LIBRARY_RT AS PRECURSOR_LIBRARY_RT,
            {library_drift_time} AS PRECURSOR_LIBRARY_DRIFT_TIME,
            {gene_id} AS GENE_ID,
            {gene_name} AS GENE_NAME,
            {gene_decoy} AS GENE_DECOY,
            PROTEIN.DECOY AS PROTEIN_DECOY,
            PEPTIDE.DECOY AS PEPTIDE_DECOY,
            PRECURSOR.DECOY AS PRECURSOR_DECOY,
            FEATURE.RUN_ID AS RUN_ID,
            RUN.FILENAME,
            FEATURE.ID AS FEATURE_ID,
            FEATURE.EXP_RT,
            {exp_im} AS EXP_IM,
            FEATURE.NORM_RT,
            FEATURE.DELTA_RT,
            FEATURE.LEFT_WIDTH,
            FEATURE.RIGHT_WIDTH,
            {feature_ms1_cols_sql},
            {feature_ms2_cols_sql},
            TRANSITION_PEPTIDE_MAPPING.PEPTIDE_ID AS IPF_PEPTIDE_ID,
            TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID AS PRECURSOR_ID,
            TRANSITION.ID AS TRANSITION_ID,
            TRANSITION.TRAML_ID AS TRANSITION_TRAML_ID,
            TRANSITION.PRODUCT_MZ,
            TRANSITION.CHARGE AS TRANSITION_CHARGE,
            TRANSITION.TYPE AS TRANSITION_TYPE,
            TRANSITION.ORDINAL AS TRANSITION_ORDINAL,
            {annotation} AS ANNOTATION,
            TRANSITION.DETECTING AS TRANSITION_DETECTING,
            TRANSITION.LIBRARY_INTENSITY AS TRANSITION_LIBRARY_INTENSITY,
            TRANSITION.DECOY AS TRANSITION_DECOY,
            {feature_transition_cols_sql}
        FROM sqlite_scan('{infile}', 'PRECURSOR') AS PRECURSOR
        INNER JOIN sqlite_scan('{infile}', 'PRECURSOR_PEPTIDE_MAPPING') AS PRECURSOR_PEPTIDE_MAPPING 
            ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        INNER JOIN sqlite_scan('{infile}', 'PEPTIDE') AS PEPTIDE 
            ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN sqlite_scan('{infile}', 'PEPTIDE_PROTEIN_MAPPING') AS PEPTIDE_PROTEIN_MAPPING 
            ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        INNER JOIN sqlite_scan('{infile}', 'PROTEIN') AS PROTEIN 
            ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
        {gene_joins}
        -- Join Precursor feature data
        INNER JOIN sqlite_scan('{infile}', 'FEATURE') AS FEATURE 
            ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN sqlite_scan('{infile}', 'FEATURE_MS1') AS FEATURE_MS1 
            ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
        INNER JOIN sqlite_scan('{infile}', 'FEATURE_MS2') AS FEATURE_MS2 
            ON FEATURE.ID = FEATURE_MS2.FEATURE_ID
        INNER JOIN sqlite_scan('{infile}', 'RUN') AS RUN 
            ON FEATURE.RUN_ID = RUN.ID
        -- Join Transition info
        INNER JOIN sqlite_scan('{infile}', 'TRANSITION_PRECURSOR_MAPPING') AS TRANSITION_PRECURSOR_MAPPING 
            ON PRECURSOR.ID = TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID
        INNER JOIN sqlite_scan('{infile}', 'TRANSITION') AS TRANSITION
            ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
        FULL JOIN sqlite_scan('{infile}', 'TRANSITION_PEPTIDE_MAPPING') AS TRANSITION_PEPTIDE_MAPPING 
            ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
        -- Join Transition feature data
        FULL JOIN sqlite_scan('{infile}', 'FEATURE_TRANSITION') AS FEATURE_TRANSITION 
            ON TRANSITION.ID = FEATURE_TRANSITION.TRANSITION_ID
        """

        query = template.format(
            infile=infile,
            library_drift_time="PRECURSOR.LIBRARY_DRIFT_TIME" if has_library_drift_time else "NULL",
            gene_id="PEPTIDE_GENE_MAPPING.GENE_ID" if gene_tables_exist else "NULL",
            gene_name="GENE.GENE_NAME" if gene_tables_exist else "NULL",
            gene_decoy="GENE.DECOY" if gene_tables_exist else "NULL",
            gene_joins=(
                "LEFT JOIN sqlite_scan('{infile}', 'PEPTIDE_GENE_MAPPING') AS PEPTIDE_GENE_MAPPING "
                "ON PEPTIDE.ID = PEPTIDE_GENE_MAPPING.PEPTIDE_ID "
                "LEFT JOIN sqlite_scan('{infile}', 'GENE') AS GENE "
                "ON PEPTIDE_GENE_MAPPING.GENE_ID = GENE.ID"
            ).format(infile=infile) if gene_tables_exist else "",
            exp_im="FEATURE.EXP_IM" if has_im else "NULL",
            feature_ms1_cols_sql=feature_ms1_cols_sql,
            feature_ms2_cols_sql=feature_ms2_cols_sql,
            annotation=(
                "TRANSITION.ANNOTATION"
                if has_annotation
                else "TRANSITION.TYPE || CAST(TRANSITION.ORDINAL AS VARCHAR) || '^' || CAST(TRANSITION.CHARGE AS VARCHAR)"
            ),
            feature_transition_cols_sql=feature_transition_cols_sql,
        )

        copy_query = f"COPY ({query}) TO '{outfile}' (FORMAT 'parquet', COMPRESSION '{compression_method}', COMPRESSION_LEVEL {compression_level});"

        conn.execute(copy_query)

    conn.close()
    

def convert_sqmass_to_parquet(
    infile, outfile, oswfile, compression_method="zstd", compression_level=11
):
    '''
    Convert a SQMass sqlite file to Parquet format
    '''
    
    xic_conn = duckdb.connect(database=infile, read_only=True)
    load_sqlite_scanner(xic_conn)
    osw_conn = duckdb.connect(database=oswfile, read_only=True)
    load_sqlite_scanner(osw_conn)

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
