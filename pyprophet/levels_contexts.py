# encoding: latin-1

import sys
import pandas as pd
pd.options.display.width = 220
pd.options.display.precision = 6

import numpy as np
import sqlite3
from .stats import error_statistics, lookup_values_from_error_table
from .report import save_report
from shutil import copyfile
from .std_logger import logging

# Protein Grouping
from collections import defaultdict as ddict

def statistics_report(data, outfile, context, analyte, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

    error_stat, pi0 = error_statistics(data[data.decoy==0]['score'], data[data.decoy==1]['score'], pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, parametric, pfdr, True, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

    p_values, s_values, peps, q_values = lookup_values_from_error_table(data["score"].values, error_stat)
    data["p_value"] = p_values
    data["s_value"] = s_values
    data["q_value"] = q_values
    data["pep"] = peps

    if context == 'run-specific':
        outfile = outfile + "_" + str(data['run_id'].unique()[0])

    # export PDF report
    save_report(outfile + "_" + context + "_" + analyte + ".pdf", outfile + ": " + context + " " + analyte + "-level error-rate control", data[data.decoy==1]["score"], data[data.decoy==0]["score"], data["score"], data["s_value"], data["q_value"], data[data.decoy==0]["p_value"], pi0)

    return(data)


def infer_proteins(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

    con = sqlite3.connect(infile)

    if context == 'global':
        data = pd.read_sql_query('SELECT NULL AS RUN_ID, PROTEIN.ID AS GROUP_ID, PROTEIN.ID AS PROTEIN_ID, PRECURSOR.DECOY, SCORE, "global" AS CONTEXT FROM PROTEIN INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID INNER JOIN PEPTIDE ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID GROUP BY GROUP_ID HAVING MAX(SCORE) ORDER BY SCORE DESC', con)

    elif context == 'run-specific':
        data = pd.read_sql_query('SELECT RUN_ID || "_" || PROTEIN.ID AS GROUP_ID, RUN_ID, PROTEIN.ID AS PROTEIN_ID, PRECURSOR.DECOY, SCORE, "run-specific" AS CONTEXT FROM PROTEIN INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID INNER JOIN PEPTIDE ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID GROUP BY GROUP_ID HAVING MAX(SCORE) ORDER BY SCORE DESC', con)

    elif context == 'experiment-wide':
        data = pd.read_sql_query('SELECT RUN_ID || "_" || PROTEIN.ID AS GROUP_ID, RUN_ID, PROTEIN.ID AS PROTEIN_ID, PRECURSOR.DECOY, SCORE, "experiment-wide" AS CONTEXT FROM PROTEIN INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID INNER JOIN PEPTIDE ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID GROUP BY GROUP_ID HAVING MAX(SCORE) ORDER BY SCORE DESC', con)

    else:
        sys.exit("Error: Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]
    con.close()

    if context == 'run-specific':
        data = data.groupby('run_id').apply(statistics_report, outfile, context, "protein", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps).reset_index()

    elif context in ['global', 'experiment-wide']:
        data = statistics_report(data, outfile, context, "protein", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

    # store data in table
    if infile != outfile:
        copyfile(infile, outfile)

    con = sqlite3.connect(outfile)

    c = con.cursor()
    c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_PROTEIN"')
    if c.fetchone()[0] == 1:
        c.execute('DELETE FROM SCORE_PROTEIN WHERE CONTEXT =="' + context + '"')
    c.fetchall()

    df = data[['context','run_id','protein_id','score','p_value','q_value','pep']]
    df.columns = ['CONTEXT','RUN_ID','PROTEIN_ID','SCORE','PVALUE','QVALUE','PEP']
    table = "SCORE_PROTEIN"
    df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')

    con.close()


# def infer_protein_groups(infile, outfile, peptide_qvalue, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):
#     import maspy.inference

#     con = sqlite3.connect(infile)

#     if context == 'global':
#         data = pd.read_sql_query('SELECT SCORE_PEPTIDE.PEPTIDE_ID, PROTEIN_ID, PROTEIN.DECOY FROM SCORE_PEPTIDE INNER JOIN PEPTIDE_PROTEIN_MAPPING ON SCORE_PEPTIDE.PEPTIDE_ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID INNER JOIN PROTEIN ON PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID WHERE qvalue < ' + str(peptide_qvalue), con)

#     else:
#         sys.exit("Error: Unspecified context selected.")
#     data.columns = [col.lower() for col in data.columns]
#     con.close()

#     # Protein Grouping
#     proteins_peptides = ddict(set)
#     for index, row in data.iterrows():
#         proteins_peptides[row['protein_id']].add(row['peptide_id'])
#     protein_groups = maspy.inference.mappingBasedGrouping(proteins_peptides)

#     group_dict = {'context': [], 'id': [], 'protein_id': []}
#     for group_id in protein_groups.groups:
#         leadingProteins = protein_groups.groups[group_id].leading
#         for protein_id in leadingProteins:
#             group_dict['context'].append(context)
#             group_dict['id'].append(group_id)
#             group_dict['protein_id'].append(protein_id)

#     df = pd.DataFrame(group_dict)

#     # store data in table
#     if infile != outfile:
#         copyfile(infile, outfile)
    
#     con = sqlite3.connect(outfile)

#     c = con.cursor()
#     c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PROTEIN_GROUPS"')
#     if c.fetchone()[0] == 1:
#         c.execute('DELETE FROM PROTEIN_GROUPS WHERE CONTEXT =="' + context + '"')
#     c.fetchall()

#     df = df[['context','id','protein_id']]
#     df.columns = ['CONTEXT','ID','PROTEIN_ID']
#     table = "PROTEIN_GROUPS"
#     df.to_sql(table, con, index=False, if_exists='append')

#     con.close()

def infer_peptides(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

    con = sqlite3.connect(infile)

    if context == 'global':
        data = pd.read_sql_query('SELECT NULL AS RUN_ID, PEPTIDE.ID AS GROUP_ID, PEPTIDE.ID AS PEPTIDE_ID, PRECURSOR.DECOY, SCORE, "global" AS CONTEXT FROM PEPTIDE INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID GROUP BY GROUP_ID HAVING MAX(SCORE) ORDER BY SCORE DESC', con)

    elif context == 'run-specific':
        data = pd.read_sql_query('SELECT RUN_ID || "_" || PEPTIDE.ID AS GROUP_ID, RUN_ID, PEPTIDE.ID AS PEPTIDE_ID, PRECURSOR.DECOY, SCORE, "run-specific" AS CONTEXT FROM PEPTIDE INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID GROUP BY GROUP_ID HAVING MAX(SCORE) ORDER BY SCORE DESC', con)

    elif context == 'experiment-wide':
        data = pd.read_sql_query('SELECT RUN_ID || "_" || PEPTIDE.ID AS GROUP_ID, RUN_ID, PEPTIDE.ID AS PEPTIDE_ID, PRECURSOR.DECOY, SCORE, "experiment-wide" AS CONTEXT FROM PEPTIDE INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID GROUP BY GROUP_ID HAVING MAX(SCORE) ORDER BY SCORE DESC', con)

    else:
        sys.exit("Error: Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]
    con.close()

    if context == 'run-specific':
        data = data.groupby('run_id').apply(statistics_report, outfile, context, "peptide", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps).reset_index()

    elif context in ['global', 'experiment-wide']:
        data = statistics_report(data, outfile, context, "peptide", parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

    # store data in table
    if infile != outfile:
        copyfile(infile, outfile)
    
    con = sqlite3.connect(outfile)

    c = con.cursor()
    c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="SCORE_PEPTIDE"')
    if c.fetchone()[0] == 1:
        c.execute('DELETE FROM SCORE_PEPTIDE WHERE CONTEXT =="' + context + '"')
    c.fetchall()

    df = data[['context','run_id','peptide_id','score','p_value','q_value','pep']]
    df.columns = ['CONTEXT','RUN_ID','PEPTIDE_ID','SCORE','PVALUE','QVALUE','PEP']
    table = "SCORE_PEPTIDE"
    df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')

    con.close()

def merge_osw(infiles, outfile, subsample_ratio, test):
    for infile in infiles:
        if infile == infiles[0]:
            # Copy the first file to have a template
            copyfile(infile, outfile)
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            c.execute('DELETE FROM RUN')
            c.execute('DELETE FROM FEATURE')
            c.execute('DELETE FROM FEATURE_MS1')
            c.execute('DELETE FROM FEATURE_MS2')
            c.execute('DELETE FROM FEATURE_TRANSITION')
            conn.commit()
            c.fetchall()
            conn.close()
            
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        c.execute('ATTACH DATABASE "'+ infile + '" AS sdb')
        c.execute('INSERT INTO RUN SELECT * FROM sdb.RUN')
        if subsample_ratio >= 1.0:
            c.execute('INSERT INTO FEATURE SELECT * FROM sdb.FEATURE')
        else:
            if not test:
                c.execute('INSERT INTO FEATURE SELECT * FROM sdb.FEATURE WHERE PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sdb.FEATURE ORDER BY RANDOM() LIMIT (SELECT ROUND(' + str(subsample_ratio) + '*COUNT(DISTINCT PRECURSOR_ID)) FROM sdb.FEATURE))')
            else:
                c.execute('INSERT INTO FEATURE SELECT * FROM sdb.FEATURE WHERE PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sdb.FEATURE LIMIT (SELECT ROUND(' + str(subsample_ratio) + '*COUNT(DISTINCT PRECURSOR_ID)) FROM sdb.FEATURE))')
        c.execute('INSERT INTO FEATURE_MS1 SELECT * FROM sdb.FEATURE_MS1 WHERE sdb.FEATURE_MS1.FEATURE_ID IN (SELECT ID FROM FEATURE)')
        c.execute('INSERT INTO FEATURE_MS2 SELECT * FROM sdb.FEATURE_MS2 WHERE sdb.FEATURE_MS2.FEATURE_ID IN (SELECT ID FROM FEATURE)')
        c.execute('INSERT INTO FEATURE_TRANSITION SELECT * FROM sdb.FEATURE_TRANSITION WHERE sdb.FEATURE_TRANSITION.FEATURE_ID IN (SELECT ID FROM FEATURE)')
        logging.info("Merging file " + infile + " to " + outfile + ".")
        conn.commit()
        c.fetchall()
        conn.close()

    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    c.execute('VACUUM')
    conn.commit()
    c.fetchall()
    conn.close()
    logging.info("All OSW files were merged.")
