import sys
import os
import click
import pandas as pd
import numpy as np
import sqlite3

from .stats import error_statistics, lookup_values_from_error_table, final_err_table, summary_err_table
from .report import save_report
from shutil import copyfile
from .data_handling import check_sqlite_table


def statistics_report(data, outfile, context, analyte, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

    error_stat, pi0 = error_statistics(data[data.decoy==0]['score'], data[data.decoy==1]['score'], parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, True, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

    stat_table = final_err_table(error_stat)
    summary_table = summary_err_table(error_stat)

    # print summary table
    click.echo("=" * 80)
    click.echo(summary_table)
    click.echo("=" * 80)

    p_values, s_values, peps, q_values = lookup_values_from_error_table(data["score"].values, error_stat)
    data["p_value"] = p_values
    data["s_value"] = s_values
    data["q_value"] = q_values
    data["pep"] = peps

    if context == 'run-specific':
        outfile = outfile + "_" + str(data['run_id'].unique()[0])

    # export PDF report
    save_report(outfile + "_" + context + "_" + analyte + ".pdf", outfile + ": " + context + " " + analyte + "-level error-rate control", data[data.decoy==1]["score"], data[data.decoy==0]["score"], stat_table["cutoff"], stat_table["svalue"], stat_table["qvalue"], data[data.decoy==0]["p_value"], pi0)

    return(data)


def infer_proteins(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

    con = sqlite3.connect(infile)

    if context in ['global','experiment-wide','run-specific']:
        if context == 'global':
            run_id = 'NULL'
        else:
            run_id = 'RUN_ID'
        data = pd.read_sql_query('''
SELECT %s AS RUN_ID,
       PROTEIN.ID AS GROUP_ID,
       PROTEIN.ID AS PROTEIN_ID,
       PRECURSOR.DECOY AS DECOY,
       SCORE,
       "%s" AS CONTEXT
FROM PROTEIN
INNER JOIN
  (SELECT PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID AS PEPTIDE_ID,
          PROTEIN_ID
   FROM
     (SELECT PEPTIDE_ID,
             COUNT(*) AS NUM_PROTEINS
      FROM PEPTIDE_PROTEIN_MAPPING
      GROUP BY PEPTIDE_ID) AS PROTEINS_PER_PEPTIDE
   INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PROTEINS_PER_PEPTIDE.PEPTIDE_ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
   WHERE NUM_PROTEINS == 1) AS PEPTIDE_PROTEIN_MAPPING ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
INNER JOIN PEPTIDE ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
GROUP BY GROUP_ID
HAVING MAX(SCORE)
ORDER BY SCORE DESC
''' % (run_id, context), con)
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
        c.execute('DELETE FROM SCORE_PROTEIN WHERE CONTEXT =="%s"' % context)
    c.fetchall()

    df = data[['context','run_id','protein_id','score','p_value','q_value','pep']]
    df.columns = ['CONTEXT','RUN_ID','PROTEIN_ID','SCORE','PVALUE','QVALUE','PEP']
    table = "SCORE_PROTEIN"
    df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')

    con.close()


def infer_peptides(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

    con = sqlite3.connect(infile)

    if context in ['global','experiment-wide','run-specific']:
        if context == 'global':
            run_id = 'NULL'
        else:
            run_id = 'RUN_ID'
        data = pd.read_sql_query('''
SELECT %s AS RUN_ID,
       PEPTIDE.ID AS GROUP_ID,
       PEPTIDE.ID AS PEPTIDE_ID,
       PRECURSOR.DECOY,
       SCORE,
       "%s" AS CONTEXT
FROM PEPTIDE
INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
INNER JOIN PRECURSOR ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
GROUP BY GROUP_ID
HAVING MAX(SCORE)
ORDER BY SCORE DESC
''' % (run_id, context), con)
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
        c.execute('DELETE FROM SCORE_PEPTIDE WHERE CONTEXT =="%s"' % context)
    c.fetchall()

    df = data[['context','run_id','peptide_id','score','p_value','q_value','pep']]
    df.columns = ['CONTEXT','RUN_ID','PEPTIDE_ID','SCORE','PVALUE','QVALUE','PEP']
    table = "SCORE_PEPTIDE"
    df.to_sql(table, con, index=False, dtype={"RUN_ID": "INTEGER"}, if_exists='append')

    con.close()


def merge_osw(infiles, outfile, subsample_ratio, test):
    # Copy the first file to have a template
    copyfile(infiles[0], outfile)
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    c.executescript('''
PRAGMA synchronous = OFF;

DELETE FROM RUN;

DELETE FROM FEATURE;

DELETE FROM FEATURE_MS1;

DELETE FROM FEATURE_MS2;

DELETE FROM FEATURE_TRANSITION;
''')

    for infile in infiles:
        c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO RUN SELECT * FROM sdb.RUN;

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Merged runs of file %s to %s." % (infile, outfile))

    for infile in infiles:
        if subsample_ratio >= 1.0:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb; 

INSERT INTO FEATURE SELECT * FROM sdb.FEATURE; 

DETACH DATABASE sdb;
''' % infile)
        else:
            if test:
                c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE
SELECT *
FROM sdb.FEATURE
WHERE PRECURSOR_ID IN
    (SELECT ID
     FROM sdb.PRECURSOR
     LIMIT
       (SELECT ROUND(%s*COUNT(DISTINCT ID))
        FROM sdb.PRECURSOR));

DETACH DATABASE sdb;
''' % (infile, subsample_ratio))
            else:
                c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE
SELECT *
FROM sdb.FEATURE
WHERE PRECURSOR_ID IN
    (SELECT ID
     FROM sdb.PRECURSOR
     ORDER BY RANDOM()
     LIMIT
       (SELECT ROUND(%s*COUNT(DISTINCT ID))
        FROM sdb.PRECURSOR));

DETACH DATABASE sdb;
''' % infile, subsample_ratio)
        click.echo("Info: Merged generic features of file %s to %s." % (infile, outfile))

    for infile in infiles:
        if subsample_ratio >= 1.0:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_MS1
SELECT *
FROM sdb.FEATURE_MS1;

DETACH DATABASE sdb;
''' % infile)
        else:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_MS1
SELECT *
FROM sdb.FEATURE_MS1
WHERE sdb.FEATURE_MS1.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Merged MS1 features of file %s to %s." % (infile, outfile))

    for infile in infiles:
        if subsample_ratio >= 1.0:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_MS2
SELECT *
FROM sdb.FEATURE_MS2;

DETACH DATABASE sdb;
''' % infile)
        else:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_MS2
SELECT *
FROM sdb.FEATURE_MS2
WHERE sdb.FEATURE_MS2.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Merged MS2 features of file %s to %s." % (infile, outfile))

    for infile in infiles:
        if subsample_ratio >= 1.0:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_TRANSITION
SELECT *
FROM sdb.FEATURE_TRANSITION;

DETACH DATABASE sdb;
''' % infile)
        else:
            c.executescript('''
ATTACH DATABASE "%s" AS sdb;

INSERT INTO FEATURE_TRANSITION
SELECT *
FROM sdb.FEATURE_TRANSITION
WHERE sdb.FEATURE_TRANSITION.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
''' % infile)
        click.echo("Info: Merged transition features of file %s to %s." % (infile, outfile))

    conn.commit()
    conn.close()

    click.echo("Info: All OSW files were merged.")


def reduce_osw(infile, outfile):
    conn = sqlite3.connect(infile)
    if not check_sqlite_table(conn, "SCORE_MS2"):
        sys.exit("Error: Apply scoring to MS2 data before reducing file for multi-run scoring.")
    conn.close()

    try:
        os.remove(outfile)
    except OSError:
        pass

    conn = sqlite3.connect(outfile)
    c = conn.cursor()

    c.executescript('''
PRAGMA synchronous = OFF;

ATTACH DATABASE "%s" AS sdb;

' + '
CREATE TABLE RUN(ID INT PRIMARY KEY NOT NULL,
                 FILENAME TEXT NOT NULL);

INSERT INTO RUN
SELECT *
FROM sdb.RUN;

CREATE TABLE SCORE_MS2(FEATURE_ID INTEGER, SCORE REAL);

INSERT INTO SCORE_MS2 (FEATURE_ID, SCORE)
SELECT FEATURE_ID,
       SCORE
FROM sdb.SCORE_MS2
WHERE RANK == 1;

CREATE TABLE FEATURE(ID INT PRIMARY KEY NOT NULL,
                     RUN_ID INT NOT NULL,
                     PRECURSOR_ID INT NOT NULL);

INSERT INTO FEATURE (ID, RUN_ID, PRECURSOR_ID)
SELECT ID,
       RUN_ID,
       PRECURSOR_ID
FROM sdb.FEATURE
WHERE ID IN
    (SELECT FEATURE_ID
     FROM SCORE_MS2);
''' % infile)

    conn.commit()
    conn.close()

    click.echo("Info: OSW file was reduced for multi-run scoring.")


def merge_oswr(infiles, outfile, templatefile):
    # Copy the template to the output file
    copyfile(templatefile, outfile)
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    c.executescript('''
PRAGMA synchronous = OFF;

DROP TABLE IF EXISTS RUN;

DROP TABLE IF EXISTS FEATURE;

DROP TABLE IF EXISTS FEATURE_MS1;

DROP TABLE IF EXISTS FEATURE_MS2;

DROP TABLE IF EXISTS FEATURE_TRANSITION;

DROP TABLE IF EXISTS SCORE_MS1;

DROP TABLE IF EXISTS SCORE_MS2;

DROP TABLE IF EXISTS SCORE_TRANSITION;

DROP TABLE IF EXISTS SCORE_PEPTIDE;

DROP TABLE IF EXISTS SCORE_PROTEIN;

DROP TABLE IF EXISTS SCORE_IPF;

CREATE TABLE RUN(ID INT PRIMARY KEY NOT NULL,
                 FILENAME TEXT NOT NULL);

CREATE TABLE SCORE_MS2(FEATURE_ID INTEGER, SCORE REAL);

CREATE TABLE FEATURE(ID INT PRIMARY KEY NOT NULL,
                     RUN_ID INT NOT NULL,
                     PRECURSOR_ID INT NOT NULL);
''')

    for infile in infiles:
        c.executescript('ATTACH DATABASE "%s" AS sdb; INSERT INTO RUN SELECT * FROM sdb.RUN; DETACH DATABASE sdb;' % infile)
        click.echo("Info: Merged runs of file %s to %s." % (infile, outfile))

    for infile in infiles:
        c.executescript('ATTACH DATABASE "%s" AS sdb; INSERT INTO FEATURE SELECT * FROM sdb.FEATURE; DETACH DATABASE sdb;' % infile)
        click.echo("Info: Merged generic features of file %s to %s." % (infile, outfile))

    for infile in infiles:
        c.executescript('ATTACH DATABASE "%s" AS sdb; INSERT INTO SCORE_MS2 SELECT * FROM sdb.SCORE_MS2; DETACH DATABASE sdb;' % infile)
        click.echo("Info: Merged MS2 scores of file %s to %s." % (infile, outfile))

    conn.commit()
    conn.close()

    click.echo("Info: All reduced OSWR files were merged.")


def backpropagate_oswr(infile, outfile, apply_scores):
    # store data in table
    if infile != outfile:
        copyfile(infile, outfile)

    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    c.executescript('''
PRAGMA synchronous = OFF;

DROP TABLE IF EXISTS SCORE_PEPTIDE;

DROP TABLE IF EXISTS SCORE_PROTEIN;

CREATE TABLE SCORE_PEPTIDE (CONTEXT TEXT, RUN_ID INTEGER, PEPTIDE_ID INTEGER, SCORE REAL, PVALUE REAL, QVALUE REAL, PEP REAL);

CREATE TABLE SCORE_PROTEIN (CONTEXT TEXT, RUN_ID INTEGER, PROTEIN_ID INTEGER, SCORE REAL, PVALUE REAL, QVALUE REAL, PEP REAL);

ATTACH DATABASE "%s" AS sdb;

INSERT INTO SCORE_PEPTIDE
SELECT *
FROM sdb.SCORE_PEPTIDE;

INSERT INTO SCORE_PROTEIN
SELECT *
FROM sdb.SCORE_PROTEIN;
''' % apply_scores)

    conn.commit()
    conn.close()

    click.echo("Info: All multi-run data was backpropagated.")
