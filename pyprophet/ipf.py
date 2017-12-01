# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import random


import pandas as pd
pd.options.display.width = 220
pd.options.display.precision = 6

import numpy as np
import scipy as sp
import sqlite3

try:
    profile
except NameError:
    def profile(fun):
        return fun

from .std_logger import logging

pd.set_option('chained_assignment',None)

def read_pyp_peakgroup_precursor(path, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring):
    con = sqlite3.connect(path)

    if not ipf_ms1_scoring and ipf_ms2_scoring:
        data = pd.read_sql_query("SELECT FEATURE.ID AS FEATURE_ID, 1 - SCORE_MS2.PEP AS MS2_PEAKGROUP_PP, NULL AS MS1_PRECURSOR_PP, 1 - SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PP FROM PRECURSOR INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID INNER JOIN (SELECT FEATURE_ID, PEP FROM SCORE_TRANSITION INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID WHERE TRANSITION.TYPE='' AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < " + str(ipf_max_peakgroup_pep) + ";", con)
    elif ipf_ms1_scoring and not ipf_ms2_scoring:
        data = pd.read_sql_query("SELECT FEATURE.ID AS FEATURE_ID, 1 - SCORE_MS2.PEP AS MS2_PEAKGROUP_PP, 1 - SCORE_MS1.PEP AS MS1_PRECURSOR_PP, NULL AS MS2_PRECURSOR_PP FROM PRECURSOR INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < " + str(ipf_max_peakgroup_pep) + ";", con)
    elif ipf_ms1_scoring and ipf_ms2_scoring:
        data = pd.read_sql_query("SELECT FEATURE.ID AS FEATURE_ID, 1 - SCORE_MS2.PEP AS MS2_PEAKGROUP_PP, 1 - SCORE_MS1.PEP AS MS1_PRECURSOR_PP, 1 - SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PP FROM PRECURSOR INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID INNER JOIN (SELECT FEATURE_ID, PEP FROM SCORE_TRANSITION INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID WHERE TRANSITION.TYPE='' AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < " + str(ipf_max_peakgroup_pep) + ";", con)
    else:
        data = pd.read_sql_query("SELECT FEATURE.ID AS FEATURE_ID, 1 - SCORE_MS2.PEP AS MS2_PEAKGROUP_PP, NULL AS MS1_PRECURSOR_PP, NULL AS MS2_PRECURSOR_PP FROM PRECURSOR INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID WHERE PRECURSOR.DECOY=0 AND SCORE_MS2.PEP < " + str(ipf_max_peakgroup_pep) + ";", con)

    data.columns = [col.lower() for col in data.columns]
    con.close()

    return data

def read_pyp_transition(path, ipf_max_transition_pep, ipf_h0):
    # Only the evidence is restricted to ipf_max_transition_pep, the peptidoform-space is complete.
    con = sqlite3.connect(path)

    # transition-level evidence
    evidence = pd.read_sql_query("SELECT FEATURE_ID, TRANSITION_ID, 1 - PEP AS POSTERIOR FROM SCORE_TRANSITION INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID WHERE TRANSITION.TYPE!='' AND TRANSITION.DECOY=0 AND PEP < " + str(ipf_max_transition_pep) + ";", con)
    evidence.columns = [col.lower() for col in evidence.columns]

    # transition-level bitmask
    bitmask = pd.read_sql_query("SELECT DISTINCT TRANSITION.ID AS TRANSITION_ID, PEPTIDE_ID, 1 AS BMASK FROM TRANSITION INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID WHERE TRANSITION.TYPE!='' AND TRANSITION.DECOY=0;", con)
    bitmask.columns = [col.lower() for col in bitmask.columns]

    # potential peptidoforms per feature
    num_peptidoforms = pd.read_sql_query("SELECT FEATURE_ID, COUNT(DISTINCT PEPTIDE_ID) AS NUM_PEPTIDOFORMS FROM SCORE_TRANSITION INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID WHERE TRANSITION.TYPE!='' AND TRANSITION.DECOY=0 GROUP BY FEATURE_ID ORDER BY FEATURE_ID;", con)
    num_peptidoforms.columns = [col.lower() for col in num_peptidoforms.columns]

    # peptidoform space per feature
    peptidoforms = pd.read_sql_query("SELECT DISTINCT FEATURE_ID, PEPTIDE_ID FROM SCORE_TRANSITION INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID WHERE TRANSITION.TYPE!='' AND TRANSITION.DECOY=0 ORDER BY FEATURE_ID;", con)
    peptidoforms.columns = [col.lower() for col in peptidoforms.columns]

    con.close()

    # add h0 to peptidoform-space if necessary
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
    data_matrix = []

    for i,tr in data.iterrows():
        if tr['ms1_precursor_pp'] <= 1.0 and tr['ms1_precursor_pp'] >= 0:
            data_matrix.append({'prior': tr['ms2_peakgroup_pp'], 'evidence': tr['ms1_precursor_pp'], 'hypothesis': True})
            data_matrix.append({'prior': 1-tr['ms2_peakgroup_pp'], 'evidence': 1-tr['ms1_precursor_pp'], 'hypothesis': False})
        if tr['ms2_precursor_pp'] <= 1.0 and tr['ms2_precursor_pp'] >= 0:
            data_matrix.append({'prior': tr['ms2_peakgroup_pp'], 'evidence': tr['ms2_precursor_pp'], 'hypothesis': True})
            data_matrix.append({'prior': 1-tr['ms2_peakgroup_pp'], 'evidence': 1-tr['ms2_precursor_pp'], 'hypothesis': False})

    # There is no evidence for a precursor so we set the evidence to 0.
    if len(data_matrix) == 0:
            data_matrix.append({'prior': tr['ms2_peakgroup_pp'], 'evidence': 0, 'hypothesis': True})
            data_matrix.append({'prior': 1-tr['ms2_peakgroup_pp'], 'evidence': 1, 'hypothesis': False})

    return pd.DataFrame(data_matrix)
    
def prepare_transition_bm(data):
    # peptide_id: -1 indicates h0, i.e. the peak group is wrong!

    # initialize priors
    data.ix[data.peptide_id != -1, 'prior'] = data.ix[data.peptide_id != -1, 'precursor_peakgroup_pp'] / data.ix[data.peptide_id != -1, 'num_peptidoforms'] # potential peptidoforms
    data.ix[data.peptide_id == -1, 'prior'] = 1 - data.ix[data.peptide_id == -1, 'precursor_peakgroup_pp'] # h0

    # set evidence
    data.ix[data.bmask == 1, 'evidence'] = data.ix[data.bmask == 1, 'posterior'] # we have evidence FOR this peptidoform or h0
    data.ix[data.bmask == 0, 'evidence'] = 1 - data.ix[data.bmask == 0, 'posterior'] # we have evidence AGAINST this peptidoform or h0

    data = data[['feature_id','prior','evidence','peptide_id']]
    data = data.rename(columns=lambda x: x.replace('peptide_id', 'hypothesis'))

    return data

def apply_bm(data):
    # compute likelihood * prior per id & hypothesis
    # all priors are identical but pandas DF multiplication requires aggregation, so we use min()
    pp_data = (data.groupby(['feature_id',"hypothesis"])["evidence"].prod() * data.groupby(['feature_id',"hypothesis"])["prior"].min()).reset_index()
    pp_data.columns = ['feature_id','hypothesis','likelihood_prior']

    # compute likelihood sum per id
    pp_data['likelihood_sum'] = pp_data.groupby('feature_id')['likelihood_prior'].transform(np.sum)

    # compute posterior hypothesis probability
    pp_data['posterior'] = pp_data['likelihood_prior'] / pp_data['likelihood_sum']

    return pp_data.fillna(value = 0)

def precursor_inference(data, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep):
    # get MS1-level precursor data
    if ipf_ms1_scoring:
        uis_ms1_precursor_data = data[data['ms1_precursor_pp'] >= 1-ipf_max_precursor_pep][['feature_id','ms1_precursor_pp']].drop_duplicates()
    else:
        uis_ms1_precursor_data = data[['feature_id']].drop_duplicates()
        uis_ms1_precursor_data['ms1_precursor_pp'] = np.nan

    # get MS2-level precursor data
    if ipf_ms2_scoring:
        uis_ms2_precursor_data = data[data['ms2_precursor_pp'] >= 1-ipf_max_precursor_pep][['feature_id','ms2_precursor_pp']].drop_duplicates()
    else:
        uis_ms2_precursor_data = data[['feature_id']].drop_duplicates()
        uis_ms2_precursor_data['ms2_precursor_pp'] = np.nan

    # get MS2-level peak group data
    uis_ms2_pg_data = data[['feature_id','ms2_peakgroup_pp']].drop_duplicates()

    if ipf_ms1_scoring or ipf_ms2_scoring:
        # merge MS1- & MS2-level precursor and peak group data
        uis_precursor_data = uis_ms2_precursor_data.merge(uis_ms1_precursor_data, on=['feature_id'], how='outer').merge(uis_ms2_pg_data, on=['feature_id'], how='outer')

        # prepare precursor-level Bayesian model
        uis_precursor_data_bm = uis_precursor_data.groupby('feature_id').apply(prepare_precursor_bm).reset_index()

        # compute posterior precursor probability
        prec_pp_data = apply_bm(uis_precursor_data_bm)
        prec_pp_data = prec_pp_data.rename(columns=lambda x: x.replace('posterior', 'precursor_peakgroup_pp'))

        inferred_precursors = prec_pp_data[prec_pp_data['hypothesis']][['feature_id','precursor_peakgroup_pp']]
    else:
        # no precursor-level data on MS1 and/or MS2 should be used; use peak group-level data
        inferred_precursors = uis_ms2_pg_data.rename(columns=lambda x: x.replace('ms2_peakgroup_pp', 'precursor_peakgroup_pp'))

    return inferred_precursors


def peptidoform_inference(uis_transition_table, uis_precursor_data):
    uis_transition_table = pd.merge(uis_transition_table, uis_precursor_data, on='feature_id')

    # compute transition posterior probabilities
    uis_transition_data_bm = prepare_transition_bm(uis_transition_table)

    # compute posterior peptidoform probability
    pp_data = apply_bm(uis_transition_data_bm)
    pp_data = pp_data.rename(columns=lambda x: x.replace('posterior', 'pf_score'))

    # compute model-based FDR
    pp_data['qvalue'] = compute_model_fdr(pp_data['pf_score'].values)
    pp_data['pep'] = 1 - pp_data['pf_score']
    pp_data['PosteriorFullPeptideName'] = pp_data['hypothesis']

    # merge precursor-level data with UIS data
    result = pp_data.merge(uis_precursor_data[['feature_id','precursor_peakgroup_pp']].drop_duplicates(), on=['feature_id'], how='inner')

    return result

def compute_model_fdr(data):
    fdrs = []
    for t in data:
        fdrs.append((1-data[data >= t]).sum() / len(data[data >= t]))
    return np.array(fdrs)

def infer_peptidoforms(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep):
    logging.info("start IPF (Inference of PeptidoForms)")

    # precursor level
    logging.info("      prepare precursor-level data")
    uis_precursor_table = read_pyp_peakgroup_precursor(infile, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring)

    logging.info("      conduct precursor-level inference")
    uis_precursor_data = precursor_inference(uis_precursor_table, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep)
    uis_precursor_data = uis_precursor_data[(uis_precursor_data['precursor_peakgroup_pp'] >= 1-ipf_max_precursor_peakgroup_pep)]

    # peptidoform level
    logging.info("      prepare peptidoform-level data")
    uis_transition_table = read_pyp_transition(infile, ipf_max_transition_pep, ipf_h0)

    logging.info("      conduct peptidoform-level inference")
    uis_peptidoform_data = peptidoform_inference(uis_transition_table, uis_precursor_data)

    logging.info("      finalize results")
    uis_peptidoform_data = uis_peptidoform_data[uis_peptidoform_data['hypothesis']!=-1][['feature_id','hypothesis','precursor_peakgroup_pp','qvalue','pep']]
    uis_peptidoform_data.columns = ['FEATURE_ID','PEPTIDE_ID','PRECURSOR_PEAKGROUP_POSTERIOR','QVALUE','PEP']

    if infile != outfile:
        copyfile(infile, outfile)

    con = sqlite3.connect(outfile)

    uis_peptidoform_data['PEPTIDE_ID'] = uis_peptidoform_data['PEPTIDE_ID'].astype(int)
    uis_peptidoform_data.to_sql("SCORE_IPF", con, index=False, if_exists='replace')
    con.close()

    logging.info("end IPF")

