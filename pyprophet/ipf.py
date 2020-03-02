import pandas as pd
import numpy as np
import scipy as sp
import sqlite3
import sys
import click

from scipy.stats import rankdata
from .data_handling import check_sqlite_table
from shutil import copyfile

def compute_model_fdr(data_in):
    data = np.asarray(data_in)

    # compute model based FDR estimates from posterior error probabilities
    order = np.argsort(data)

    ranks = np.zeros(data.shape[0], dtype=np.int)
    fdr = np.zeros(data.shape[0])

    # rank data with with maximum ranks for ties
    ranks[order] = rankdata(data[order], method='max')

    # compute FDR/q-value by using cumulative sum of maximum rank for ties
    fdr[order] = data[order].cumsum()[ranks[order]-1] / ranks[order]

    return fdr


def read_pyp_peakgroup_precursor(path, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring):
    click.echo("Info: Reading precursor-level data.")
    # precursors are restricted according to ipf_max_peakgroup_pep to exclude very poor peak groups
    con = sqlite3.connect(path)

    # only use MS2 precursors
    if not ipf_ms1_scoring and ipf_ms2_scoring:
        if not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(con, "SCORE_TRANSITION"):
            raise click.ClickException("Apply scoring to MS2 and transition-level data before running IPF.")

        con.executescript('''
CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_feature_id ON SCORE_TRANSITION (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id ON SCORE_TRANSITION (TRANSITION_ID);
''')

        data = pd.read_sql_query('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
       NULL AS MS1_PRECURSOR_PEP,
       SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN
  (SELECT FEATURE_ID,
          PEP
   FROM SCORE_TRANSITION
   INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
   WHERE TRANSITION.TYPE=''
     AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
WHERE PRECURSOR.DECOY=0
  AND SCORE_MS2.PEP < %s;
''' % ipf_max_peakgroup_pep, con)

    # only use MS1 precursors
    elif ipf_ms1_scoring and not ipf_ms2_scoring:
        if not check_sqlite_table(con, "SCORE_MS1") or not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(con, "SCORE_TRANSITION"):
            raise click.ClickException("Apply scoring to MS1, MS2 and transition-level data before running IPF.")

        con.executescript('''
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
CREATE INDEX IF NOT EXISTS idx_score_ms1_feature_id ON SCORE_MS1 (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
''')

        data = pd.read_sql_query('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
       SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
       NULL AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
WHERE PRECURSOR.DECOY=0
  AND SCORE_MS2.PEP < %s;
''' % ipf_max_peakgroup_pep, con)

    # use both MS1 and MS2 precursors
    elif ipf_ms1_scoring and ipf_ms2_scoring:
        if not check_sqlite_table(con, "SCORE_MS1") or not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(con, "SCORE_TRANSITION"):
            raise click.ClickException("Apply scoring to MS1, MS2 and transition-level data before running IPF.")

        con.executescript('''
CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
CREATE INDEX IF NOT EXISTS idx_score_ms1_feature_id ON SCORE_MS1 (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_feature_id ON SCORE_TRANSITION (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id ON SCORE_TRANSITION (TRANSITION_ID);
''')

        data = pd.read_sql_query('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
       SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
       SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN
  (SELECT FEATURE_ID,
          PEP
   FROM SCORE_TRANSITION
   INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
   WHERE TRANSITION.TYPE=''
     AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
WHERE PRECURSOR.DECOY=0
  AND SCORE_MS2.PEP < %s;
''' % ipf_max_peakgroup_pep, con)

    # do not use any precursor information
    else:
        if not check_sqlite_table(con, "SCORE_MS2") or not check_sqlite_table(con, "SCORE_TRANSITION"):
            raise click.ClickException("Apply scoring to MS2  and transition-level data before running IPF.")

        con.executescript('''
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
''')

        data = pd.read_sql_query('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
       NULL AS MS1_PRECURSOR_PEP,
       NULL AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
WHERE PRECURSOR.DECOY=0
  AND SCORE_MS2.PEP < %s;
''' % ipf_max_peakgroup_pep, con)

    data.columns = [col.lower() for col in data.columns]
    con.close()

    return data


def read_pyp_transition(path, ipf_max_transition_pep, ipf_h0):
    click.echo("Info: Reading peptidoform-level data.")
    # only the evidence is restricted to ipf_max_transition_pep, the peptidoform-space is complete
    con = sqlite3.connect(path)

    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_transition_peptide_mapping_transition_id ON TRANSITION_PEPTIDE_MAPPING (TRANSITION_ID);
CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_feature_id ON SCORE_TRANSITION (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id ON SCORE_TRANSITION (TRANSITION_ID);
''')

    # transition-level evidence
    evidence = pd.read_sql_query('''
SELECT FEATURE_ID,
       TRANSITION_ID,
       PEP
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
WHERE TRANSITION.TYPE!=''
  AND TRANSITION.DECOY=0
  AND PEP < %s;
 ''' % ipf_max_transition_pep, con)
    evidence.columns = [col.lower() for col in evidence.columns]

    # transition-level bitmask
    bitmask = pd.read_sql_query('''
SELECT DISTINCT TRANSITION.ID AS TRANSITION_ID,
                PEPTIDE_ID,
                1 AS BMASK
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
WHERE TRANSITION.TYPE!=''
  AND TRANSITION.DECOY=0;
''', con)
    bitmask.columns = [col.lower() for col in bitmask.columns]

    # potential peptidoforms per feature
    num_peptidoforms = pd.read_sql_query('''
SELECT FEATURE_ID,
       COUNT(DISTINCT PEPTIDE_ID) AS NUM_PEPTIDOFORMS
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
WHERE TRANSITION.TYPE!=''
  AND TRANSITION.DECOY=0
GROUP BY FEATURE_ID
ORDER BY FEATURE_ID;
''', con)
    num_peptidoforms.columns = [col.lower() for col in num_peptidoforms.columns]

    # peptidoform space per feature
    peptidoforms = pd.read_sql_query('''
SELECT DISTINCT FEATURE_ID,
                PEPTIDE_ID
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
WHERE TRANSITION.TYPE!=''
  AND TRANSITION.DECOY=0
ORDER BY FEATURE_ID;
''', con)
    peptidoforms.columns = [col.lower() for col in peptidoforms.columns]

    con.close()

    # add h0 (peptide_id: -1) to peptidoform-space if necessary
    if ipf_h0:
        peptidoforms = pd.concat([peptidoforms, pd.DataFrame({'feature_id': peptidoforms['feature_id'].unique(), 'peptide_id': -1})])

    # generate transition-peptidoform table
    trans_pf = pd.merge(evidence, peptidoforms, how='outer', on='feature_id')

    # apply bitmask
    trans_pf_bm = pd.merge(trans_pf, bitmask, how='left', on=['transition_id','peptide_id']).fillna(0)

    # append number of peptidoforms
    data = pd.merge(trans_pf_bm, num_peptidoforms, how='inner', on='feature_id')

    return data


def prepare_precursor_bm(data):
    # MS1-level precursors
    ms1_precursor_data = data[['feature_id','ms2_peakgroup_pep','ms1_precursor_pep']].dropna(axis=0, how='any')
    ms1_bm_data = pd.concat([pd.DataFrame({'feature_id': ms1_precursor_data['feature_id'], 'prior': 1-ms1_precursor_data['ms2_peakgroup_pep'], 'evidence': 1-ms1_precursor_data['ms1_precursor_pep'], 'hypothesis': True}), pd.DataFrame({'feature_id': ms1_precursor_data['feature_id'], 'prior': ms1_precursor_data['ms2_peakgroup_pep'], 'evidence': ms1_precursor_data['ms1_precursor_pep'], 'hypothesis': False})])

    # MS2-level precursors
    ms2_precursor_data = data[['feature_id','ms2_peakgroup_pep','ms2_precursor_pep']].dropna(axis=0, how='any')
    ms2_bm_data = pd.concat([pd.DataFrame({'feature_id': ms2_precursor_data['feature_id'], 'prior': 1-ms2_precursor_data['ms2_peakgroup_pep'], 'evidence': 1-ms2_precursor_data['ms2_precursor_pep'], 'hypothesis': True}), pd.DataFrame({'feature_id': ms2_precursor_data['feature_id'], 'prior': ms2_precursor_data['ms2_peakgroup_pep'], 'evidence': ms2_precursor_data['ms2_precursor_pep'], 'hypothesis': False})])

    # missing precursor data
    missing_precursor_data = data[['feature_id','ms2_peakgroup_pep']].dropna(axis=0, how='any').drop_duplicates()
    missing_bm_data = pd.concat([pd.DataFrame({'feature_id': missing_precursor_data['feature_id'], 'prior': 1-missing_precursor_data['ms2_peakgroup_pep'], 'evidence': 0, 'hypothesis': True}), pd.DataFrame({'feature_id': missing_precursor_data['feature_id'], 'prior': missing_precursor_data['ms2_peakgroup_pep'], 'evidence': 1, 'hypothesis': False})])

    # combine precursor data
    precursor_bm_data = pd.concat([ms1_bm_data, ms2_bm_data])
    # append missing precursors if no MS1/MS2 evidence is available
    precursor_bm_data = pd.concat([precursor_bm_data, missing_bm_data.loc[~missing_bm_data['feature_id'].isin(precursor_bm_data['feature_id'])]])

    return(precursor_bm_data)


def prepare_transition_bm(data):
    # peptide_id = -1 indicates h0, i.e. the peak group is wrong!
    # initialize priors
    data.loc[data.peptide_id != -1, 'prior'] = (1-data.loc[data.peptide_id != -1, 'precursor_peakgroup_pep']) / data.loc[data.peptide_id != -1, 'num_peptidoforms'] # potential peptidoforms
    data.loc[data.peptide_id == -1, 'prior'] = data.loc[data.peptide_id == -1, 'precursor_peakgroup_pep'] # h0

    # set evidence
    data.loc[data.bmask == 1, 'evidence'] = (1-data.loc[data.bmask == 1, 'pep']) # we have evidence FOR this peptidoform or h0
    data.loc[data.bmask == 0, 'evidence'] = data.loc[data.bmask == 0, 'pep'] # we have evidence AGAINST this peptidoform or h0

    data = data[['feature_id','num_peptidoforms','prior','evidence','peptide_id']]
    data = data.rename(columns=lambda x: x.replace('peptide_id', 'hypothesis'))

    return data


def apply_bm(data):
    # compute likelihood * prior per feature & hypothesis
    # all priors are identical but pandas DF multiplication requires aggregation, so we use min()
    pp_data = (data.groupby(['feature_id',"hypothesis"])["evidence"].prod() * data.groupby(['feature_id',"hypothesis"])["prior"].min()).reset_index()
    pp_data.columns = ['feature_id','hypothesis','likelihood_prior']

    # compute likelihood sum per feature
    pp_data['likelihood_sum'] = pp_data.groupby('feature_id')['likelihood_prior'].transform(np.sum)

    # compute posterior hypothesis probability
    pp_data['posterior'] = pp_data['likelihood_prior'] / pp_data['likelihood_sum']

    return pp_data.fillna(value = 0)


def precursor_inference(data, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep, ipf_max_precursor_peakgroup_pep):
    # prepare MS1-level precursor data
    if ipf_ms1_scoring:
        ms1_precursor_data = data[data['ms1_precursor_pep'] < ipf_max_precursor_pep][['feature_id','ms1_precursor_pep']].drop_duplicates()
    else:
        ms1_precursor_data = data[['feature_id']].drop_duplicates()
        ms1_precursor_data['ms1_precursor_pep'] = np.nan

    # prepare MS2-level precursor data
    if ipf_ms2_scoring:
        ms2_precursor_data = data[data['ms2_precursor_pep'] < ipf_max_precursor_pep][['feature_id','ms2_precursor_pep']].drop_duplicates()
    else:
        ms2_precursor_data = data[['feature_id']].drop_duplicates()
        ms2_precursor_data['ms2_precursor_pep'] = np.nan

    # prepare MS2-level peak group data
    ms2_pg_data = data[['feature_id','ms2_peakgroup_pep']].drop_duplicates()

    if ipf_ms1_scoring or ipf_ms2_scoring:
        # merge MS1- & MS2-level precursor and peak group data
        precursor_data = ms2_precursor_data.merge(ms1_precursor_data, on=['feature_id'], how='outer').merge(ms2_pg_data, on=['feature_id'], how='outer')

        # prepare precursor-level Bayesian model
        click.echo("Info: Preparing precursor-level data.")
        precursor_data_bm = prepare_precursor_bm(precursor_data)

        # compute posterior precursor probability
        click.echo("Info: Conducting precursor-level inference.")
        prec_pp_data = apply_bm(precursor_data_bm)
        prec_pp_data['precursor_peakgroup_pep'] = 1 - prec_pp_data['posterior']

        inferred_precursors = prec_pp_data[prec_pp_data['hypothesis']][['feature_id','precursor_peakgroup_pep']]
    else:
        # no precursor-level data on MS1 and/or MS2 should be used; use peak group-level data
        click.echo("Info: Skipping precursor-level inference.")
        inferred_precursors = ms2_pg_data.rename(columns=lambda x: x.replace('ms2_peakgroup_pep', 'precursor_peakgroup_pep'))

    inferred_precursors = inferred_precursors[(inferred_precursors['precursor_peakgroup_pep'] < ipf_max_precursor_peakgroup_pep)]

    return inferred_precursors


def peptidoform_inference(transition_table, precursor_data, ipf_grouped_fdr):
    transition_table = pd.merge(transition_table, precursor_data, on='feature_id')

    # compute transition posterior probabilities
    click.echo("Info: Preparing peptidoform-level data.")
    transition_data_bm = prepare_transition_bm(transition_table)

    # compute posterior peptidoform probability
    click.echo("Info: Conducting peptidoform-level inference.")
    pf_pp_data = apply_bm(transition_data_bm)
    pf_pp_data['pep'] = 1 - pf_pp_data['posterior']

    # compute model-based FDR
    if ipf_grouped_fdr:
        pf_pp_data['qvalue'] = pd.merge(pf_pp_data, transition_data_bm[['feature_id', 'num_peptidoforms']].drop_duplicates(), on=['feature_id'], how='inner').groupby('num_peptidoforms')['pep'].transform(compute_model_fdr)
    else:
        pf_pp_data['qvalue'] = compute_model_fdr(pf_pp_data['pep'])

    # merge precursor-level data with UIS data
    result = pf_pp_data.merge(precursor_data[['feature_id','precursor_peakgroup_pep']].drop_duplicates(), on=['feature_id'], how='inner')

    return result


def infer_peptidoforms(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_grouped_fdr, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep):
    click.echo("Info: Starting IPF (Inference of PeptidoForms).")

    # precursor level
    precursor_table = read_pyp_peakgroup_precursor(infile, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring)
    precursor_data = precursor_inference(precursor_table, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep, ipf_max_precursor_peakgroup_pep)

    # peptidoform level
    peptidoform_table = read_pyp_transition(infile, ipf_max_transition_pep, ipf_h0)
    peptidoform_data = peptidoform_inference(peptidoform_table, precursor_data, ipf_grouped_fdr)

    # finalize results and write to table
    click.echo("Info: Storing results.")
    peptidoform_data = peptidoform_data[peptidoform_data['hypothesis']!=-1][['feature_id','hypothesis','precursor_peakgroup_pep','qvalue','pep']]
    peptidoform_data.columns = ['FEATURE_ID','PEPTIDE_ID','PRECURSOR_PEAKGROUP_PEP','QVALUE','PEP']

    if infile != outfile:
        copyfile(infile, outfile)

    con = sqlite3.connect(outfile)

    peptidoform_data.to_sql("SCORE_IPF", con, index=False, if_exists='replace')
    con.close()
