import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sqlite3
import argparse
from .data_handling import check_sqlite_table
from datetime import datetime

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()
    export_parquet(args.infile, args.outfile, 1)

# this method is only currently supported for combined output and not with ipf
def export_to_parquet(infile, outfile, transitionLevel):

    con = sqlite3.connect(infile)

    print("Creating Index Query ...")
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

    print("Creating Index ....")

    con.executescript(idx_query) # Add indices

    # create transition indicies (if needed)
    if transitionLevel:
        idx_transition_query = '''
        CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);
        CREATE INDEX IF NOT EXISTS idx_transition_transition_id ON TRANSITION (ID);
        CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id_feature_id ON FEATURE_TRANSITION (TRANSITION_ID, FEATURE_ID);
        CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID); '''

        print("Creating transition level index ...")
        con.executescript(idx_transition_query)

    
    print("Creating Main Query ....")
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
        print("[INFO] Ion Mobility Columns Found")
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

    # query, start with the largest table and work way outwards 
    # Start with feature_transition but then to include the rows of those precursors not found (no associated feature) join with precursor_transition
    if transitionLevel:
        query = '''
        SELECT {0}
        FROM FEATURE_TRANSITION

        LEFT JOIN FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
        LEFT JOIN FEATURE_MS1 ON FEATURE_TRANSITION.FEATURE_ID = FEATURE_MS1.FEATURE_ID
        LEFT JOIN FEATURE_MS2 ON FEATURE_TRANSITION.FEATURE_ID = FEATURE_MS2.FEATURE_ID

        LEFT JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
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

        SELECT DISTINCT {0} FROM TRANSITION_PRECURSOR_MAPPING

        LEFT JOIN PRECURSOR ON TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID = PRECURSOR.ID
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

        '''.format(columnsToSelect)

    else:
        query = '''
        SELECT {0}
        FROM FEATURE

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


        UNION

        SELECT DISTINCT {0} FROM PRECURSOR

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

        '''.format(columnsToSelect)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Executing Query (Current Time = {})".format(current_time))
    
    try:
        print("Attempting Execution")
        df = pd.read_sql(query, con)
        print("read_sql_query went well")
    except Exception as e:
        print("read_sql_query failed: "+ str(e))


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Done Executing (Current Time = {})".format(current_time))

    print("Creating bitwise maps ...")

    # create masks for easier indexing 
    df['FEATURE_MASK'] = (~ df['FEATURE_ID'].duplicated()) & (df['FEATURE_ID'].notna())

    # for the precursor mask should be a superset of feature mask. If precursor mask is true and feature mask is false then feature should be NA.
    # Thus df[(~df['FEATURE_MASK']) & df['PRECURSOR_MASK'] & df['FEATURE_ID'].notna()] should return no entries
    # in current implementation this is not directly looked for however because all of the transitions for a given precursor/feature are groupped together in the array it works out.
    df['PRECURSOR_MASK'] = (~ df['PRECURSOR_ID'].duplicated())


    df['PEPTIDE_MASK'] = ~ df['PEPTIDE_ID'].duplicated()

    print("Saving metaData ...")

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


    fixed_table = table.replace_schema_metadata({**custom_metadata, **existing_metadata})

    merged_metadata = { **custom_metadata, **existing_metadata }
    fixed_table = table.replace_schema_metadata(merged_metadata)
    
    con.close()
    print("Saving to Parquet .... ")

    ## export to parquet 
    pq.write_table(fixed_table, outfile) 


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Done Saving (Current Time = {})".format(current_time))


if __name__ == "__main__":
    main()