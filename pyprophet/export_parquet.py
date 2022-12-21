import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sqlite3
import click
from .data_handling import check_sqlite_table
from datetime import datetime
from tqdm import tqdm
import os
import multiprocessing
from functools import wraps
import contextlib
from time import time

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

def read_precursor_feature_data(con, columnsToSelect, prec_ids):
            query = f'''
        SELECT DISTINCT {columnsToSelect} 
        FROM (SELECT * FROM PRECURSOR WHERE ID in ({','.join(prec_ids)})) AS PRECURSOR

            LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            LEFT JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            LEFT JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
            LEFT JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID

            LEFT JOIN FEATURE ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            LEFT JOIN FEATURE_MS1 ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
            LEFT JOIN FEATURE_MS2 ON FEATURE.ID = FEATURE_MS2.FEATURE_ID

            LEFT JOIN RUN ON FEATURE.RUN_ID = RUN.ID

            LEFT JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
            LEFT JOIN SCORE_PEPTIDE ON PEPTIDE.ID = SCORE_PEPTIDE.PEPTIDE_ID
            LEFT JOIN SCORE_PROTEIN ON PROTEIN.ID = SCORE_PROTEIN.PROTEIN_ID

            WHERE FEATURE.ID IS NULL
        
        UNION

        SELECT {columnsToSelect}
        FROM (SELECT * FROM FEATURE WHERE PRECURSOR_ID in ({','.join(prec_ids)})) AS FEATURE

            LEFT JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
            LEFT JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
            LEFT JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
            LEFT JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID

            LEFT JOIN RUN ON FEATURE.RUN_ID = RUN.ID
            LEFT JOIN FEATURE_MS1 ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
            LEFT JOIN FEATURE_MS2 ON FEATURE.ID = FEATURE_MS2.FEATURE_ID


            LEFT JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
            LEFT JOIN SCORE_PEPTIDE ON PEPTIDE.ID = SCORE_PEPTIDE.PEPTIDE_ID
            LEFT JOIN SCORE_PROTEIN ON PROTEIN.ID = SCORE_PROTEIN.PROTEIN_ID
        '''
            df = pd.read_sql(query, con)
            return df

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

def osw_to_parquet_writer(con, columnsToSelect, precursor_id_batches, outfile, osw_data_reader=read_precursor_feature_data):
    # If an input file is passed instead of a sqlite3 connection, establish a connection
    if isinstance(con, sqlite3.Connection):
        con = sqlite3.connect(con)
    writer = None
    for prec_id in tqdm(precursor_id_batches, desc="INFO: Reading data from OSW...", total=len(precursor_id_batches)):
        df = osw_data_reader(con, columnsToSelect, prec_id['ID'].astype(str).values)
        writer = append_to_parquet_table(df, outfile, writer)
    if writer:
        writer.close()

# this method is only currently supported for combined output and not with ipf
@method_timer
def export_to_parquet(infile, outfile, transitionLevel, chunksize=1000, threads=1):
    '''
    Convert an OSW sqlite file to Parquet format

    Parameters:
        infile: (str) path to osw sqlite file
        outfile: (str) path to write out parquet file
        transitionLevel: (bool) append transition level data
        chunksize: (int) read in the data into chunks for low-memory requirements
    
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

CREATE INDEX IF NOT EXISTS idx_score_protein_protein_id ON SCORE_PROTEIN (PROTEIN_ID);
CREATE INDEX IF NOT EXISTS idx_score_protein_run_id ON SCORE_PROTEIN (RUN_ID);

CREATE INDEX IF NOT EXISTS idx_score_peptide_peptide_id ON SCORE_PEPTIDE (PEPTIDE_ID);
CREATE INDEX IF NOT EXISTS idx_score_peptide_run_id ON SCORE_PEPTIDE (RUN_ID);
'''

    if check_sqlite_table(con, "FEATURE_MS1"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);"
    if check_sqlite_table(con, "FEATURE_MS2"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);"
    if check_sqlite_table(con, "SCORE_MS2"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);"

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

    ## features
    columns['FEATURE'] = ['RUN_ID', 'EXP_RT', 'EXP_IM', 'NORM_RT', 'DELTA_RT', 'LEFT_WIDTH', 'RIGHT_WIDTH']
    columns['FEATURE_MS2'] = ['AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI', 'VAR_BSERIES_SCORE', 'VAR_DOTPROD_SCORE', 'VAR_INTENSITY_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE', 'VAR_LIBRARY_CORR', 'VAR_LIBRARY_DOTPROD', 'VAR_LIBRARY_MANHATTAN', 'VAR_LIBRARY_RMSD', 'VAR_LIBRARY_ROOTMEANSQUARE', 'VAR_LIBRARY_SANGLE', 'VAR_LOG_SN_SCORE', 'VAR_MANHATTAN_SCORE', 'VAR_MASSDEV_SCORE', 'VAR_MASSDEV_SCORE_WEIGHTED', 'VAR_MI_SCORE', 'VAR_MI_WEIGHTED_SCORE', 'VAR_MI_RATIO_SCORE', 'VAR_NORM_RT_SCORE', 'VAR_XCORR_COELUTION', 'VAR_XCORR_COELUTION_WEIGHTED', 'VAR_XCORR_SHAPE', 'VAR_XCORR_SHAPE_WEIGHTED', 'VAR_YSERIES_SCORE', 'VAR_ELUTION_MODEL_FIT_SCORE', 'VAR_IM_XCORR_SHAPE', 'VAR_IM_XCORR_COELUTION', 'VAR_IM_DELTA_SCORE', 'VAR_SONAR_LAG', 'VAR_SONAR_SHAPE', 'VAR_SONAR_LOG_SN', 'VAR_SONAR_LOG_DIFF', 'VAR_SONAR_LOG_TREND', 'VAR_SONAR_RSQ']
    columns['FEATURE_MS1'] = ['AREA_INTENSITY', 'APEX_INTENSITY', 'VAR_MASSDEV_SCORE', 'VAR_MI_SCORE', 'VAR_MI_CONTRAST_SCORE', 'VAR_MI_COMBINED_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE', 'VAR_IM_MS1_DELTA_SCORE', 'VAR_XCORR_COELUTION', 'VAR_XCORR_COELUTION_CONTRAST', 'VAR_XCORR_COELUTION_COMBINED', 'VAR_XCORR_SHAPE', 'VAR_XCORR_SHAPE_CONTRAST', 'VAR_XCORR_SHAPE_COMBINED']

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
    columns['SCORE_PEPTIDE'] = ['CONTEXT', 'SCORE', 'PVALUE', 'QVALUE', 'PEP']
    columns['SCORE_PROTEIN'] = ['SCORE', 'PVALUE', 'QVALUE', 'PEP']

    ## other
    columns['RUN'] = ['FILENAME']

    ## mappings
    columns['PRECURSOR_PEPTIDE_MAPPING'] = ['PEPTIDE_ID', 'PRECURSOR_ID']
    #columns['TRANSITION_PRECURSOR_MAPPING'] = ['PRECURSOR_ID']
    columns['PEPTIDE_PROTEIN_MAPPING'] = ['PROTEIN_ID']

    # transition level
    if transitionLevel:
        columns['FEATURE_TRANSITION'] = ['FEATURE_ID', 'TRANSITION_ID', 'AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI', 'VAR_INTENSITY_SCORE', 'VAR_INTENSITY_RATIO_SCORE', 'VAR_LOG_INTENSITY', 'VAR_XCORR_COELUTION', 'VAR_XCORR_SHAPE', 'VAR_LOG_SN_SCORE', 'VAR_MASSDEV_SCORE', 'VAR_MI_SCORE', 'VAR_MI_RATIO_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE']
        columns['TRANSITION'] = ['TRAML_ID', 'PRODUCT_MZ', 'CHARGE', 'TYPE', 'ORDINAL', 'DETECTING', 'IDENTIFYING', 'QUANTIFYING', 'LIBRARY_INTENSITY']
    else:
        columns['FEATURE'].append('FEATURE.ID as FEATURE_ID')

    ### rename column names that are in common 
    whitelist = ['PEPTIDE_ID', 'FEATURE_ID', 'TRANSITION_ID', 'PRECURSOR_ID', 'PROTEIN_ID', 'DECOY']  # these columns should not be renamed
    for c in columns.keys():
        for i in range(len(columns[c])):
            # if not in whitelist add the table name to the column name
            if 'as' in columns[c][i]: # do not rename if already added custom name ("indicated by as", note for these entries need to specify the column name
                columns[c][i] = columns[c][i]
            elif not columns[c][i] in whitelist:
                columns[c][i] = '{0}.{1} as "{0}.{1}"'.format(c, columns[c][i])
            else:
                columns[c][i] = "{0}.{1} as {1}".format(c, columns[c][i])

    ## convert the columns to a string for sql query 
    columnsToSelect = ''
    for c in columns.keys():
        columnsToSelect += ", ".join(columns[c])
        columnsToSelect += ", "
    
    #get rid of trailing comma
    columnsToSelect = columnsToSelect[0:-2]

    # Get list of precursor ids
    precursor_ids = pd.read_sql("SELECT ID FROM PRECURSOR", con)
    precursor_id_batches = [precursor_ids[i:i+chunksize].copy() for i in range(0,precursor_ids.shape[0],chunksize)]

    # Start with feature_transition but then to include the rows of those precursors not found (no associated feature) join with precursor_transition
    if transitionLevel: # each row will be a transition 
        osw_data_reader = read_feature_transition_data
    else: # each row will be a precursor/feature
        osw_data_reader = read_precursor_feature_data

    with code_block_timer(f"Info: Extracting data from OSW file..."):
        if threads == 1:
            osw_to_parquet_writer(con, columnsToSelect, precursor_id_batches, outfile, osw_data_reader=osw_data_reader)
            df = pd.read_parquet(outfile, engine='pyarrow')
            # Remove parquet, as this is not the final saved parquet
            os.remove(outfile) 
        else:
            # Silience VisibleDeprecationWarning
            # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            tmp_outfiles = [f"{os.path.dirname(outfile)}{os.sep}tmp_{thread}_{os.path.basename(outfile)}" for thread in range(0, threads)]
            # Initiate a pool with nthreads for parallel processing
            pool = multiprocessing.Pool( threads )
            _ = pool.starmap( osw_to_parquet_writer, zip([infile] * threads, [columnsToSelect] * threads, np.array_split(precursor_id_batches, threads), tmp_outfiles, [osw_data_reader] * threads) )
            pool.close()
            pool.join()
            # Merge tmp parquets
            df = pd.concat(map(pd.read_parquet, tmp_outfiles))
            # Remove tmp parquets
            _ = list(map(os.remove, tmp_outfiles))

    # create masks for easier data exploration
    click.echo("Info: Creating bitwise maps ...")

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

    click.echo("Info: Saving metaData ...")

    table = pa.Table.from_pandas(df)
    
    # array to store metadata 
    custom_metadata = {}
    existing_metadata = table.schema.metadata


    # fetch the OSW version as metadata
    custom_metadata['version'] = str(con.execute("select id from version").fetchone()[0])

    # fetch the pyprophet weights if avaliable
    if check_sqlite_table(con, "PYPROPHET_WEIGHTS"):
        custom_metadata['scoreLevel'] = str(con.execute("select level from PYPROPHET_WEIGHTS").fetchone()[0])
        custom_metadata['pyprophetWeights'] = pd.read_sql("select * from pyprophet_weights", con).to_json()

    # fetch the pyprophet weights if avaliable
    if check_sqlite_table(con, "PYPROPHET_XGB"):
        custom_metadata['xgbModel'] = con.execute(("select * from PYPROPHET_XGB").fetchone()[0])

    fixed_table = table.replace_schema_metadata({**custom_metadata, **existing_metadata})

    merged_metadata = { **custom_metadata, **existing_metadata }
    fixed_table = table.replace_schema_metadata(merged_metadata)
    
    con.close()
    click.echo("Info: Saving to Parquet .... ")

    ## export to parquet 
    pq.write_table(fixed_table, outfile) 

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    click.echo("Info: Done Saving (Current Time = {})".format(current_time))
