import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sqlite3
import click
from .data_handling import check_sqlite_table, to_valid_filename
from datetime import datetime
from tqdm import tqdm
import os
from pathlib import Path
import multiprocessing
from functools import wraps
import contextlib
from time import time
import re
import warnings

def method_timer(f):
    """
    Method for timing functions
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        click.echo('Info: method:%r args:[%r, %r] took: %2.4f sec' % (
            f.__name__, args, kw, te-ts))
        return result
    return wrap


@contextlib.contextmanager
def code_block_timer(ident, log_type=click.echo):
    """
    Time a block of code
    """
    tstart = time()
    yield
    elapsed = time() - tstart
    log_type("{0}: Elapsed {1} ms".format(ident, elapsed))

def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

# See: https://stackoverflow.com/questions/47113813/using-pyarrow-how-do-you-append-to-parquet-file
def append_to_parquet_table(dataframe, filepath=None, writer=None):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.

    :param dataframe: pd.DataFrame to be written in parquet format.
    :param filepath: target file location for parquet file.
    :param writer: ParquetWriter object to write pyarrow tables in parquet format.
    :return: ParquetWriter object. This can be passed in the subsequenct method calls to append DataFrame
        in the pyarrow Table
    """
    table = pa.Table.from_pandas(dataframe)
    if writer is None:
        writer = pq.ParquetWriter(filepath, table.schema)
    writer.write_table(table=table)
    return writer


def get_context_joins(columnsToSelect, run_id):
    '''
    Get join statements for peptide and protein scores and the contexts for building sql query

    Parameters:
        columnsToSelect:    (str)   string of columns to select and return in main sql query
        run_id: (int)   run_id from RUN table to filter query
    
    Return:
        context_statements will contain a string of LEFT JOIN statements
    '''
    context_table_mapping = {
        "SCORE_PEPTIDE-RUN_SPECIFIC": "SCORE_PEPTIDE",
        "SCORE_PEPTIDE-EXPERIMENT_WIDE": "SCORE_PEPTIDE",
            "SCORE_PEPTIDE-GLOBAL": "SCORE_PEPTIDE",
            "SCORE_PROTEIN-RUN_SPECIFIC": "SCORE_PROTEIN",
            "SCORE_PROTEIN-EXPERIMENT_WIDE": "SCORE_PROTEIN",
            "SCORE_PROTEIN-GLOBAL": "SCORE_PROTEIN"
    }
    join_statements = []
    for column, table_name in context_table_mapping.items():
        if re.search(column.replace("-", "_"), columnsToSelect):
            context = column.split("-")[-1].upper()
            alias = f"{table_name}_{context}"
            join_statements.append(
                f"LEFT JOIN (SELECT * FROM {table_name} WHERE RUN_ID = {run_id} AND CONTEXT = '{context}') AS {alias} ON {table_name.replace('SCORE_', '')}.ID = {alias}.{table_name.replace('SCORE_', '')}_ID")
    context_statements = "\n".join(join_statements)
    return context_statements

def get_context_joins(columnsToSelect, run_id, pep_prot_dict):
    '''
    Get join statements for peptide and protein scores and the contexts for building sql query

    Parameters:
        columnsToSelect:    (str)   string of columns to select and return in main sql query
        run_id: (int)   run_id from RUN table to filter query
    
    Return:
        context_statements will contain a string of LEFT JOIN statements
    '''
    context_table_mapping = {
        "SCORE_PEPTIDE-RUN_SPECIFIC": "SCORE_PEPTIDE",
        "SCORE_PEPTIDE-EXPERIMENT_WIDE": "SCORE_PEPTIDE",
            "SCORE_PEPTIDE-GLOBAL": "SCORE_PEPTIDE",
            "SCORE_PROTEIN-RUN_SPECIFIC": "SCORE_PROTEIN",
            "SCORE_PROTEIN-EXPERIMENT_WIDE": "SCORE_PROTEIN",
            "SCORE_PROTEIN-GLOBAL": "SCORE_PROTEIN"
    }
    join_statements = []
    for column, table_name in context_table_mapping.items():
        if re.search(column.replace("-", "_"), columnsToSelect):
            context = column.split("-")[-1].upper()
            alias = f"{table_name}_{context}"
            context = column.split("-")[-1].replace('_', '-').lower()
            join_statements.append(
                f"LEFT JOIN (SELECT * FROM {table_name} WHERE RUN_ID = {run_id} AND CONTEXT = '{context}' AND {table_name.replace('SCORE_', '')}_ID IN ({pep_prot_dict[table_name.replace('SCORE_', '')+'_ID']})) AS {alias} ON RUN.ID = {alias}.RUN_ID")
    context_statements = "\n".join(join_statements)
    return context_statements

def get_context_joins(columnsToSelect, run_id, pep_prot_dict):
    '''
    Get join statements for peptide and protein scores and the contexts for building sql query

    Parameters:
        columnsToSelect:    (str)   string of columns to select and return in main sql query
        run_id: (int)   run_id from RUN table to filter query
    
    Return:
        context_statements will contain a string of LEFT JOIN statements
    '''
    context_tables = ['SCORE_PEPTIDE', 'SCORE_PROTEIN']
    join_statements = []
    for table_name in context_tables:
        if re.search(table_name, columnsToSelect):
            join_statements.append(
                f"LEFT JOIN (SELECT * FROM {table_name} WHERE RUN_ID = {run_id} AND {table_name.replace('SCORE_', '')}_ID IN ({pep_prot_dict[table_name.replace('SCORE_', '')+'_ID']})) AS {table_name} ON RUN.ID = {table_name}.RUN_ID")
    context_statements = "\n".join(join_statements)
    return context_statements

def read_precursor_feature_data(con, columnsToSelect, prec_ids, run_id, outfile):

        # Feature Data
        feature_query = f'''
        SELECT {','.join([col.strip() for col in columnsToSelect.split(',') if re.search('^FEATURE|^RUN|^SCORE_MS2', col.strip())] + ['FEATURE.PRECURSOR_ID'])}
        FROM (SELECT * FROM FEATURE WHERE PRECURSOR_ID IN ({','.join(prec_ids)}) AND RUN_ID = {run_id}) AS FEATURE
        LEFT JOIN (SELECT * FROM RUN WHERE ID = {run_id}) AS RUN ON FEATURE.RUN_ID = RUN.ID
        LEFT JOIN FEATURE_MS1 ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
        LEFT JOIN FEATURE_MS2 ON FEATURE.ID = FEATURE_MS2.FEATURE_ID
        LEFT JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        '''
        # Read into Pandas Dataframe
        df_feature = pd.read_sql(feature_query, con)

        # Precursor Library Data
        ## Check if Gene tables are present
        if check_sqlite_table(con, "GENE"):
            gene_table_joins = '''
            LEFT JOIN PEPTIDE_GENE_MAPPING ON PEPTIDE.ID = PEPTIDE_GENE_MAPPING.GENE_ID
            LEFT JOIN GENE ON PEPTIDE_GENE_MAPPING.GENE_ID = GENE.ID
            '''
        else:
            gene_table_joins = ''

        precursor_query = f'''
        SELECT {','.join([col.strip() for col in columnsToSelect.split(',') if re.search('^PRECURSOR|^PRECURSOR_PEPTIDE_MAPPING|^PEPTIDE|^PEPTIDE_PROTEIN_MAPPING|^PROTEIN', col.strip())])}
        FROM (SELECT * FROM PRECURSOR WHERE ID in ({','.join(prec_ids)})) AS PRECURSOR
        LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        LEFT JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        LEFT JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        LEFT JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
        {gene_table_joins}
        '''
        # Read into Pandas Dataframe
        df_prec = pd.read_sql(precursor_query, con)

        # Merge Feature and Precursor Tables
        df_tmp = pd.merge(df_feature, df_prec, how='outer', on=['PRECURSOR_ID'])
        # Fill in run if for cases where precursors have no features identified
        df_tmp['FEATURE.RUN_ID'] = df_tmp['FEATURE.RUN_ID'].dropna().unique()[0]

        # Check to see if GENE_ID is in table, and check to see if all values are NULL
        if 'GENE_ID' in df_tmp.columns:
            # There are cases where a precursor doesn't have a GENE_ID mapping, which results in an nan, and a NULL in the schema, which causes a conflict
            df_tmp["GENE_ID"] = df_tmp["GENE_ID"].fillna(-1).astype('int64')

        # Check for Inferrence Context Scores
        if check_sqlite_table(con, "SCORE_PEPTIDE"):
            df_peptide_scores = pd.read_sql(f'''
            SELECT * 
            FROM SCORE_PEPTIDE
            WHERE RUN_ID = {run_id} 
            AND PEPTIDE_ID in ({','.join(df_prec.PEPTIDE_ID.astype(str).values.tolist())})
            ''', con)
            # Check if there are no scores for current precursor batch, if so, then fill in dummy data to satisfy constant schema
            if df_peptide_scores.shape[0]==0:
                # Get contexts that are in the OSW
                df_peptide_contexts = pd.read_sql(f'SELECT DISTINCT CONTEXT FROM SCORE_PEPTIDE', con)
                df_peptide_scores['CONTEXT'] = df_peptide_contexts 
            df_peptide_scores_wide = df_peptide_scores.pivot(index=['RUN_ID', 'PEPTIDE_ID'], columns='CONTEXT')
            df_peptide_scores_wide.columns = ['SCORE_PEPTIDE.' + col.upper() for col in df_peptide_scores_wide.columns.map('_'.join)]
            df_peptide_scores_wide = df_peptide_scores_wide.reset_index()
            df_tmp = pd.merge(df_tmp, df_peptide_scores_wide, how='left', left_on=['PEPTIDE_ID', 'FEATURE.RUN_ID'], right_on=['PEPTIDE_ID', 'RUN_ID'])
            df_tmp.drop(columns='RUN_ID', inplace=True)
        
        if check_sqlite_table(con, "SCORE_PROTEIN"):
            df_protein_scores = pd.read_sql(f'''
            SELECT * 
            FROM SCORE_PROTEIN
            WHERE RUN_ID = {run_id} 
            AND PROTEIN_ID in ({','.join(df_prec.PROTEIN_ID.astype(str).values.tolist())})
            ''', con)
            # Check if there are no scores for current precursor batch, if so, then fill in dummy data to satisfy constant schema
            if df_protein_scores.shape[0]==0:
                # Get contexts that are in the OSW
                df_protein_contexts = pd.read_sql(f'SELECT DISTINCT CONTEXT FROM SCORE_PROTEIN', con)
                df_protein_scores['CONTEXT'] = df_protein_contexts 
            df_protein_scores_wide = df_protein_scores.pivot(index=['RUN_ID', 'PROTEIN_ID'], columns='CONTEXT')
            df_protein_scores_wide.columns = ['SCORE_PROTEIN.' + col.upper() for col in df_protein_scores_wide.columns.map('_'.join)]
            df_protein_scores_wide = df_protein_scores_wide.reset_index()
            df_tmp = pd.merge(df_tmp, df_protein_scores_wide, how='left', left_on=['PROTEIN_ID', 'FEATURE.RUN_ID'], right_on=['PROTEIN_ID', 'RUN_ID'])
            df_tmp.drop(columns='RUN_ID', inplace=True)

        if check_sqlite_table(con, "SCORE_GENE"):
            df_gene_scores = pd.read_sql(f'''
            SELECT * 
            FROM SCORE_GENE
            WHERE RUN_ID = {run_id} 
            AND GENE_ID in ({','.join(df_prec.GENE_ID.astype(str).values.tolist())})
            ''', con)
            # Check if there are no scores for current precursor batch, if so, then fill in dummy data to satisfy constant schema
            if df_gene_scores.shape[0]==0:
                # Get contexts that are in the OSW
                df_gene_contexts = pd.read_sql(f'SELECT DISTINCT CONTEXT FROM SCORE_GENE', con)
                df_gene_scores['CONTEXT'] = df_gene_contexts 
            df_gene_scores_wide = df_gene_scores.pivot(index=['RUN_ID', 'GENE_ID'], columns='CONTEXT')
            df_gene_scores_wide.columns = ['SCORE_GENE.' + col.upper() for col in df_gene_scores_wide.columns.map('_'.join)]
            df_gene_scores_wide = df_gene_scores_wide.reset_index()
            df_tmp = pd.merge(df_tmp, df_gene_scores_wide, how='left', left_on=['GENE_ID', 'FEATURE.RUN_ID'], right_on=['GENE_ID', 'RUN_ID'])
            df_tmp.drop(columns='RUN_ID', inplace=True)

        return df_tmp

def read_feature_transition_data(con, columnsToSelect, prec_ids):
    query = f'''
        SELECT {columnsToSelect}
        FROM FEATURE_TRANSITION

        LEFT JOIN FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
        LEFT JOIN FEATURE_MS1 ON FEATURE_TRANSITION.FEATURE_ID = FEATURE_MS1.FEATURE_ID
        LEFT JOIN FEATURE_MS2 ON FEATURE_TRANSITION.FEATURE_ID = FEATURE_MS2.FEATURE_ID

        LEFT JOIN (SELECT * FROM PRECURSOR WHERE ID in ({','.join(prec_ids)})) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        LEFT JOIN TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
        LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        LEFT JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        LEFT JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        LEFT JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID

        LEFT JOIN RUN ON FEATURE.RUN_ID = RUN.ID

        LEFT JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        LEFT JOIN SCORE_PEPTIDE ON PEPTIDE.ID = SCORE_PEPTIDE.PEPTIDE_ID
        LEFT JOIN SCORE_PROTEIN ON PROTEIN.ID = SCORE_PROTEIN.PROTEIN_ID


        UNION

        SELECT DISTINCT {columnsToSelect} FROM TRANSITION_PRECURSOR_MAPPING

        LEFT JOIN (SELECT * FROM PRECURSOR WHERE ID in ({','.join(prec_ids)})) AS PRECURSOR ON TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        LEFT JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID

        LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        LEFT JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        LEFT JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        LEFT JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID

        LEFT JOIN FEATURE_TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = FEATURE_TRANSITION.TRANSITION_ID

        LEFT JOIN FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
        LEFT JOIN FEATURE_MS1 ON FEATURE_TRANSITION.FEATURE_ID = FEATURE_MS1.FEATURE_ID
        LEFT JOIN FEATURE_MS2 ON FEATURE_TRANSITION.FEATURE_ID = FEATURE_MS2.FEATURE_ID

        LEFT JOIN RUN ON FEATURE.RUN_ID = RUN.ID

        LEFT JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
        LEFT JOIN SCORE_PEPTIDE ON PEPTIDE.ID = SCORE_PEPTIDE.PEPTIDE_ID
        LEFT JOIN SCORE_PROTEIN ON PROTEIN.ID = SCORE_PROTEIN.PROTEIN_ID

        WHERE FEATURE.ID IS NULL

        '''
    df = pd.read_sql(query, con)
    return df

def append_bit_masks(outfile):
    # Read data from data written to parquet file
    df = pd.read_parquet(outfile, engine='pyarrow')
    ### FEATURE_MASK such that each row is a feature. For feature level this will remove rows of precursors that do not map to any feature. For transition level a random transition is chosen
    df['FEATURE_MASK'] = (~df['FEATURE_ID'].duplicated(keep='first')) & (df['FEATURE_ID'].notna())
    #### TOP_FEATURE_MASK = all FEATURES that have a RANK == 1. One row per feature
    df['TOP_FEATURE_MASK'] = (df['FEATURE_MASK']) & (df['SCORE_MS2.RANK'] == 1)  # just take a random transitionLevel

    #### PRECURSOR_MASK = one row per precursor, includes precursors that do not map to any feature. If precursor maps to a feature than take the feature with rank == 1. If no rank values exist mask includes a random feature.

    df.sort_values(by='SCORE_MS2.RANK', inplace=True)
    df['PRECURSOR_MASK'] = (df['TOP_FEATURE_MASK'] | df['FEATURE_ID'].isna() | df['SCORE_MS2.RANK'].isna() ) 

    dfPrec = df[df['PRECURSOR_MASK']]
    # df['PRECURSOR_MASK'] = ~dfPrec['PRECURSOR_ID'].duplicated(keep='first') ## TODO: This seems to be slow as well
    tmp = ~dfPrec['PRECURSOR_ID'].duplicated(keep='first')
    df.iloc[tmp.index, np.where(df.columns=='PRECURSOR_MASK')[0]] = tmp.values
    df['PRECURSOR_MASK'] = df['PRECURSOR_MASK'].fillna(False) ## since df and dfPrec are diferent sizes have some NA values, fill those that with False 

    table = pa.Table.from_pandas(df)
    # Write out to parquet file with bitmasks
    pq.write_table(table, outfile) 

def append_metadata(con, outfile):
    # Read data from data written to parquet file
    df = pd.read_parquet(outfile, engine='pyarrow')
    table = pa.Table.from_pandas(df)
    
    # array to store metadata 
    custom_metadata = {}
    existing_metadata = table.schema.metadata

    custom_metadata['version'] = str(con.execute("select id from version").fetchone()[0])

    # fetch the pyprophet weights if avaliable
    if check_sqlite_table(con, "PYPROPHET_WEIGHTS"):
        custom_metadata['scoreLevel'] = str(con.execute("select level from PYPROPHET_WEIGHTS").fetchone()[0])
        custom_metadata['pyprophetWeights'] = pd.read_sql("select * from pyprophet_weights", con).to_json()

    # fetch the pyprophet weights if avaliable
    if check_sqlite_table(con, "PYPROPHET_XGB"):
        custom_metadata['xgbModel'] = con.execute(("select * from PYPROPHET_XGB")).fetchone()[0]

    fixed_table = table.replace_schema_metadata({**custom_metadata, **existing_metadata})

    ## export to parquet 
    pq.write_table(fixed_table, outfile) 

def osw_to_parquet_writer(con, columnsToSelect, precursor_id_batches, run_ids, outfile, osw_data_reader=read_precursor_feature_data):
    # If an input file is passed instead of a sqlite3 connection, establish a connection
    if not isinstance(con, sqlite3.Connection):
        con = sqlite3.connect(con)
    if isinstance(outfile, str):
        writer = None
        for run_id in run_ids:
            for prec_id in tqdm(precursor_id_batches, desc=f"INFO: Reading data from OSW for run {run_id} with batch {len(precursor_id_batches)} precursor ids to file {outfile}...", total=len(precursor_id_batches)):
                df = osw_data_reader(con, columnsToSelect, prec_id['ID'].astype(str).values, run_id, outfile)
                writer = append_to_parquet_table(df, outfile, writer)
        if writer:
            writer.close()
            
            # create masks for easier data exploration
            # click.echo("Info: Creating bitwise maps ...")
            append_bit_masks(outfile)
               
            # Append metadata
            # click.echo("Info: Saving metaData ...")
            append_metadata(con, outfile)
    else:
        for run_id, out_file in zip(run_ids, outfile):
            writer = None
            for prec_id in tqdm(precursor_id_batches, desc=f"INFO: Reading data from OSW for run {run_id} with batch {len(precursor_id_batches)} precursor ids to file {out_file}...", total=len(precursor_id_batches)):
                df = osw_data_reader(con, columnsToSelect, prec_id['ID'].astype(str).values, run_id, out_file)
                writer = append_to_parquet_table(df, out_file, writer)
            if writer:
                writer.close()
            # create masks for easier data exploration
            # click.echo("Info: Creating bitwise maps ...")
            append_bit_masks(out_file)
            # Append metadata
            # click.echo("Info: Saving metaData ...")
            append_metadata(con, out_file)


# this method is only currently supported for combined output and not with ipf
@method_timer
def export_to_parquet(infile, outfile, transitionLevel, separate_runs=True, chunksize=1000, threads=1):
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
    con = sqlite3.connect(infile)

    click.echo("Info: Creating Index Query ...")
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
    

    click.echo("Info: Creating Index ....")

    con.executescript(idx_query) # Add indices

    # create transition indicies (if needed)
    if transitionLevel:
        idx_transition_query = '''
        CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);
        CREATE INDEX IF NOT EXISTS idx_transition_transition_id ON TRANSITION (ID);
        CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id_feature_id ON FEATURE_TRANSITION (TRANSITION_ID, FEATURE_ID);
        CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID); '''

        click.echo("Info: Creating transition level index ...")
        con.executescript(idx_transition_query)

    
    click.echo("Info: Creating Main Query ....")
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
    columns['FEATURE_MS2'] = ['AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI', 'VAR_BSERIES_SCORE', 'VAR_DOTPROD_SCORE', 'VAR_INTENSITY_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE', 'VAR_LIBRARY_CORR', 'VAR_LIBRARY_DOTPROD', 'VAR_LIBRARY_MANHATTAN', 'VAR_LIBRARY_RMSD', 'VAR_LIBRARY_ROOTMEANSQUARE', 'VAR_LIBRARY_SANGLE', 'VAR_LOG_SN_SCORE', 'VAR_MANHATTAN_SCORE', 'VAR_MASSDEV_SCORE', 'VAR_MASSDEV_SCORE_WEIGHTED', 'VAR_MI_SCORE', 'VAR_MI_WEIGHTED_SCORE', 'VAR_MI_RATIO_SCORE', 'VAR_NORM_RT_SCORE', 'VAR_XCORR_COELUTION', 'VAR_XCORR_COELUTION_WEIGHTED', 'VAR_XCORR_SHAPE', 'VAR_XCORR_SHAPE_WEIGHTED', 'VAR_YSERIES_SCORE', 'VAR_ELUTION_MODEL_FIT_SCORE', 'VAR_IM_XCORR_SHAPE', 'VAR_IM_XCORR_COELUTION', 'VAR_IM_DELTA_SCORE', 'VAR_SONAR_LAG', 'VAR_SONAR_SHAPE', 'VAR_SONAR_LOG_SN', 'VAR_SONAR_LOG_DIFF', 'VAR_SONAR_LOG_TREND', 'VAR_SONAR_RSQ']
    columns['FEATURE_MS1'] = ['APEX_INTENSITY', 'VAR_MASSDEV_SCORE', 'VAR_MI_SCORE', 'VAR_MI_CONTRAST_SCORE', 'VAR_MI_COMBINED_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE', 'VAR_IM_MS1_DELTA_SCORE', 'VAR_XCORR_COELUTION', 'VAR_XCORR_COELUTION_CONTRAST', 'VAR_XCORR_COELUTION_COMBINED', 'VAR_XCORR_SHAPE', 'VAR_XCORR_SHAPE_CONTRAST', 'VAR_XCORR_SHAPE_COMBINED']

    # check if IM columns exist
    query = con.execute("select count(*) as cntrec from pragma_table_info('feature_MS2') where name='EXP_IM'")
    hasIm = query.fetchone()[0] > 0
    if hasIm:
        click.echo("Info: Ion Mobility Columns Found")
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
        columns['FEATURE_TRANSITION'] = ['FEATURE_ID', 'TRANSITION_ID', 'AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI', 'VAR_INTENSITY_SCORE', 'VAR_INTENSITY_RATIO_SCORE', 'VAR_LOG_INTENSITY', 'VAR_XCORR_COELUTION', 'VAR_XCORR_SHAPE', 'VAR_LOG_SN_SCORE', 'VAR_MASSDEV_SCORE', 'VAR_MI_SCORE', 'VAR_MI_RATIO_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE']
        columns['TRANSITION'] = ['TRAML_ID', 'PRODUCT_MZ', 'CHARGE', 'TYPE', 'ORDINAL', 'DETECTING', 'IDENTIFYING', 'QUANTIFYING', 'LIBRARY_INTENSITY']
    else:
        columns['FEATURE'].append('FEATURE.ID AS FEATURE_ID')
    

    ### rename column names that are in common 
    whitelist = set(['PEPTIDE_ID', 'FEATURE_ID', 'TRANSITION_ID', 'PRECURSOR_ID', 'PROTEIN_ID', 'GENE_ID', 'DECOY'])  # these columns should not be renamed
    renamed_columns = {c: [col if 'AS' in col  else f"{c}.{col} AS '{c}.{col}'" if col not in whitelist else f"{c}.{col} AS '{col}'" for col in columns[c]] for c in columns.keys()}

    # create a list of all the columns
    columns_list = [col for c in renamed_columns.values() for col in c]

    # join the list into a single string separated by a comma and a space
    columnsToSelect = ", ".join(columns_list)

    # Get list of precursor ids
    precursor_ids = pd.read_sql("SELECT ID FROM PRECURSOR", con)
    # Shuffle precursor ids for batch processing to avoid cases where schema might change
    precursor_ids = precursor_ids.sample(frac=1)
    precursor_id_batches = [precursor_ids[i:i+chunksize].copy() for i in range(0,precursor_ids.shape[0],chunksize)]

    run_table = pd.read_sql("SELECT * FROM RUN", con)
    # Remove extension to just get filename. Note: Use Path.stem twice here to remove instances of compressed filename i.e. mzML.gz
    run_table.FILENAME = run_table.FILENAME.apply(lambda x: Path(Path(x).stem).stem).apply(to_valid_filename)
    run_ids = run_table['ID'].to_numpy().flatten()
    if separate_runs and run_ids.shape[0]>1:
        outfile = [outfile+".parquet" for outfile in run_table.FILENAME]
        tmp='\n'.join(outfile) # f-string can't handle backslashes...
        click.echo(f"Info: There is more than one run in the input osw, and `separate_runs` is set to True, so will generate {run_ids.shape[0]} separate paqrquet files per run.\n{tmp}")
        del tmp
        
    click.echo(f"Info: There are {run_ids.shape[0]} runs and {precursor_ids.shape[0]} precursor ids split into {len(precursor_id_batches)} batches...")

    # Start with feature_transition but then to include the rows of those precursors not found (no associated feature) join with precursor_transition
    if transitionLevel: # each row will be a transition 
        osw_data_reader = read_feature_transition_data
    else: # each row will be a precursor/feature
        osw_data_reader = read_precursor_feature_data

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    click.echo("Info: Start Exporting to Parquet (Current Time = {})".format(current_time))

    with code_block_timer(f"Info: Extracting data from OSW file..."):
        if threads == 1: # Single Thread Processing
            osw_to_parquet_writer(con, columnsToSelect, precursor_id_batches, run_ids, outfile, osw_data_reader=osw_data_reader)
            con.close()
        elif threads > 1 and len(run_ids) > 1 and isinstance(outfile, list): # Parallel process runs and save individual parquets
            # Close connection to database, since each thread will establish it's own connection
            con.close()
            # Silience VisibleDeprecationWarning
            # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
            warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            neval = len(outfile)
            # Initiate a pool with nthreads for parallel processing
            pool = multiprocessing.Pool(processes=threads)
            start=0
            end=threads
            while neval:
                remaining = max(0, neval - threads)
                todo = neval - remaining
                neval -= todo
                todo_outfiles = outfile[start:end]
                todo_run_ids = run_ids[start:end].tolist()
                _ = pool.starmap( osw_to_parquet_writer, zip([infile] * todo, [columnsToSelect] * todo, [precursor_id_batches] * todo, [[id] for id in todo_run_ids], todo_outfiles, [osw_data_reader] * todo) )
                start+=todo
                end+=todo
            pool.close()
            pool.join()
        elif threads > 1 and len(run_ids) > 1 and isinstance(outfile, str): # Parallel process runs and save a single parquet with all runs
            # Close connection to database, since each thread will establish it's own connection
            con.close()
            # Silience VisibleDeprecationWarning
            # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
            warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            neval = len(run_ids)
            tmp_outfiles = [f"{os.path.dirname(outfile)}{os.sep}tmp_{run}_{os.path.basename(outfile)}" for run in range(0, len(run_ids))]
            # Initiate a pool with nthreads for parallel processing
            pool = multiprocessing.Pool(processes=threads)
            start=0
            end=threads
            while neval:
                remaining = max(0, neval - threads)
                todo = neval - remaining
                neval -= todo
                todo_outfiles = tmp_outfiles[start:end]
                todo_run_ids = run_ids[start:end].tolist()
                _ = pool.starmap( osw_to_parquet_writer, zip([infile] * todo, [columnsToSelect] * todo, [precursor_id_batches] * todo, [[id] for id in todo_run_ids], todo_outfiles, [osw_data_reader] * todo) )
                start+=todo
                end+=todo
            pool.close()
            pool.join()
            df = pd.concat(map(pd.read_parquet, tmp_outfiles))
            # Remove tmp parquets
            _ = list(map(os.remove, tmp_outfiles))
            table = pa.Table.from_pandas(df)
            # Write out to parquet file with bitmasks
            pq.write_table(table, outfile)
        elif threads > 1 and len(run_ids) == 1 and isinstance(outfile, str): # Parallel process single file, parallelize on precursor id batches
            # Close connection to database, since each thread will establish it's own connection
            con.close()
            # Silience VisibleDeprecationWarning
            # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
            warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            tmp_outfiles = [f"{os.path.dirname(outfile)}{os.sep}tmp_{thread}_{os.path.basename(outfile)}" for thread in range(0, threads)]
            # Initiate a pool with nthreads for parallel processing
            pool = multiprocessing.Pool(processes=threads)
            _ = pool.starmap( osw_to_parquet_writer, zip([infile] * threads, [columnsToSelect] * threads, list(chunks(precursor_id_batches, threads)), [[id] for id in run_ids.tolist() * threads], tmp_outfiles, [osw_data_reader] * threads) )
            pool.close()
            pool.join()
            df = pd.concat(map(pd.read_parquet, tmp_outfiles))
            # Remove tmp parquets
            _ = list(map(os.remove, tmp_outfiles))
            table = pa.Table.from_pandas(df)
            # Write out to parquet file with bitmasks
            pq.write_table(table, outfile)
 

    con.close()
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    click.echo("Info: Done Exporting to Parquet (Current Time = {})".format(current_time))
