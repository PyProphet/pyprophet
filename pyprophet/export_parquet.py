import pandas as pd
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
def export_to_parquet(infile, outfile): 

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
'''
    if check_sqlite_table(con, "FEATURE_MS1"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);"
    if check_sqlite_table(con, "FEATURE_MS2"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);"
    if check_sqlite_table(con, "SCORE_MS2"):
      idx_query += "CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);"


    print("Creating Index ....")

    con.executescript(idx_query) # Add indices
    
    print("Creating Main Query ....")
    # since do not want all of the columns (some columns are twice per table) manually select the columns want in a list, (note do not want decoy)
    # note TRAML_ID for precursor and transition are not the same
    columns = {}
    ## library 
    columns['PRECURSOR'] = ['TRAML_ID', 'GROUP_LABEL', 'PRECURSOR_MZ', 'CHARGE', 'LIBRARY_INTENSITY', 'LIBRARY_RT', 'LIBRARY_DRIFT_TIME']
    columns['TRANSITION'] = ['TRAML_ID', 'PRODUCT_MZ', 'CHARGE', 'TYPE', 'ORDINAL', 'DETECTING', 'IDENTIFYING', 'QUANTIFYING', 'LIBRARY_INTENSITY']
    columns['PEPTIDE'] = ['UNMODIFIED_SEQUENCE', 'MODIFIED_SEQUENCE']
    columns['PROTEIN'] = ['PROTEIN_ACCESSION']

    ## features
    columns['FEATURE'] = ['RUN_ID', 'EXP_RT', 'EXP_IM', 'NORM_RT', 'DELTA_RT', 'LEFT_WIDTH', 'RIGHT_WIDTH']
    columns['FEATURE_MS2'] = ['AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI', 'VAR_BSERIES_SCORE', 'VAR_DOTPROD_SCORE', 'VAR_INTENSITY_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE', 'VAR_LIBRARY_CORR', 'VAR_LIBRARY_DOTPROD', 'VAR_LIBRARY_MANHATTAN', 'VAR_LIBRARY_RMSD', 'VAR_LIBRARY_ROOTMEANSQUARE', 'VAR_LIBRARY_SANGLE', 'VAR_LOG_SN_SCORE', 'VAR_MANHATTAN_SCORE', 'VAR_MASSDEV_SCORE', 'VAR_MASSDEV_SCORE_WEIGHTED', 'VAR_MI_SCORE', 'VAR_MI_WEIGHTED_SCORE', 'VAR_MI_RATIO_SCORE', 'VAR_NORM_RT_SCORE', 'VAR_XCORR_COELUTION', 'VAR_XCORR_COELUTION_WEIGHTED', 'VAR_XCORR_SHAPE', 'VAR_XCORR_SHAPE_WEIGHTED', 'VAR_YSERIES_SCORE', 'VAR_ELUTION_MODEL_FIT_SCORE', 'VAR_IM_XCORR_SHAPE', 'VAR_IM_XCORR_COELUTION', 'VAR_IM_DELTA_SCORE', 'VAR_SONAR_LAG', 'VAR_SONAR_SHAPE', 'VAR_SONAR_LOG_SN', 'VAR_SONAR_LOG_DIFF', 'VAR_SONAR_LOG_TREND', 'VAR_SONAR_RSQ']
    columns['FEATURE_MS1'] = ['AREA_INTENSITY', 'APEX_INTENSITY', 'VAR_MASSDEV_SCORE', 'VAR_MI_SCORE', 'VAR_MI_CONTRAST_SCORE', 'VAR_MI_COMBINED_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE', 'VAR_IM_MS1_DELTA_SCORE', 'VAR_XCORR_COELUTION', 'VAR_XCORR_COELUTION_CONTRAST', 'VAR_XCORR_COELUTION_COMBINED', 'VAR_XCORR_SHAPE', 'VAR_XCORR_SHAPE_CONTRAST', 'VAR_XCORR_SHAPE_COMBINED']

    ### pyprophet scores 
    columns['SCORE_MS2'] = ["SCORE", "RANK", "PVALUE", "QVALUE", "PEP"]
    columns['SCORE_PEPTIDE'] = ['CONTEXT', 'SCORE', 'PVALUE', 'QVALUE', 'PEP']
    columns['SCORE_PROTEIN'] = ['SCORE', 'PVALUE', 'QVALUE', 'PEP']

    ## other
    columns['RUN'] = ['FILENAME']


    ## mappings 
    columns['PRECURSOR_PEPTIDE_MAPPING'] = ['PEPTIDE_ID', 'PRECURSOR_ID']
    #columns['TRANSITION_PRECURSOR_MAPPING'] = ['PRECURSOR_ID']
    columns['FEATURE_TRANSITION'] = ['FEATURE_ID', 'TRANSITION_ID', 'AREA_INTENSITY', 'TOTAL_AREA_INTENSITY', 'APEX_INTENSITY', 'TOTAL_MI', 'VAR_INTENSITY_SCORE', 'VAR_INTENSITY_RATIO_SCORE', 'VAR_LOG_INTENSITY', 'VAR_XCORR_COELUTION', 'VAR_XCORR_SHAPE', 'VAR_LOG_SN_SCORE', 'VAR_MASSDEV_SCORE', 'VAR_MI_SCORE', 'VAR_MI_RATIO_SCORE', 'VAR_ISOTOPE_CORRELATION_SCORE', 'VAR_ISOTOPE_OVERLAP_SCORE']
    columns['PEPTIDE_PROTEIN_MAPPING'] = ['PROTEIN_ID']

    ### rename column names that are in common 
    whitelist = ['PEPTIDE_ID', 'FEATURE_ID', 'TRANSITION_ID', 'PRECURSOR_ID', 'PROTEIN_ID']  # these columns should not be renamed
    for c in columns.keys():
        for i in range(len(columns[c])):
            # if not in whitelist add the table name to the column name
            if not columns[c][i] in whitelist:
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

    ## query
    #query = '''
    #SELECT {} 
    #FROM PRECURSOR

    #INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
    #INNER JOIN PEPTIDE ON PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
    #INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
    #INNER JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID
    #INNER JOIN TRANSITION_PRECURSOR_MAPPING ON PRECURSOR.ID = TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID
    #INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID

    #LEFT JOIN FEATURE_TRANSITION ON TRANSITION.ID = FEATURE_TRANSITION.TRANSITION_ID
    #LEFT JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
    #LEFT JOIN FEATURE_MS1 ON FEATURE.ID = FEATURE_MS1.FEATURE_ID
    #LEFT JOIN FEATURE_MS2 ON FEATURE.ID = FEATURE_MS2.FEATURE_ID

    #INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID

    #LEFT JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
    #LEFT JOIN SCORE_PEPTIDE ON PEPTIDE.ID = SCORE_PEPTIDE.PEPTIDE_ID
    #LEFT JOIN SCORE_PROTEIN ON PROTEIN.ID = SCORE_PROTEIN.PROTEIN_ID
    #WHERE SCORE_MS2.QVALUE < {} '''.format(columnsToSelect, max_q)

    # query, start with the largest table and work way outwords. Start with feature_transition but then to include the rows of those precursors not found join with precursor_transition rows should not exceed total rows in feature_transition if joins executed correctly.
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

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Executing Query (Current Time = {})".format(current_time))
    
    df = pd.read_sql(query, con)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Done Executing (Current Time = {})".format(current_time))

    con.close()

    print("Creating bitwise maps ...")

    # create masks for easier indexing 
    df['FEATURE_MASK'] = (~ df['FEATURE_ID'].duplicated()) & (df['FEATURE_ID'].notna())

    # for the precursor mask should be a superset of feature mask. If precursor mask is true and feature mask is false then feature should be NA.
    # Thus df[(~df['FEATURE_MASK']) & df['PRECURSOR_MASK'] & df['FEATURE_ID'].notna()] should return no entries
    # do not think that this is directly looked for however because all of the transitions are organized together in the array (all in one block) it works out.
    df['PRECURSOR_MASK'] = (~ df['PRECURSOR_ID'].duplicated())


    df['PEPTIDE_MASK'] = ~ df['PEPTIDE_ID'].duplicated()

    
    print("Saving to Parquet .... ")
    ## export to parquet 
    df.to_parquet(outfile)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Done Saving (Current Time = {})".format(current_time))


if __name__ == "__main__":
    main()
