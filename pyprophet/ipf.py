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

def read_pyp_peakgroup_precursor(path):
    con = sqlite3.connect(path)

    data = pd.read_sql_query("SELECT FEATURE.ID AS FEATURE_ID, 1 - SCORE_MS2.PEP AS MS2_PEAKGROUP_PP, 1 - SCORE_MS1.PEP AS MS1_PRECURSOR_PP, 1 - SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PP FROM PRECURSOR INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID INNER JOIN SCORE_MS2 ON SCORE_MS1.FEATURE_ID = SCORE_MS2.FEATURE_ID INNER JOIN (SELECT FEATURE_ID, PEP FROM SCORE_TRANSITION INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID WHERE TRANSITION.TYPE='' AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON SCORE_MS1.FEATURE_ID = SCORE_TRANSITION.FEATURE_ID WHERE PRECURSOR.DECOY=0;", con)

    data.columns = [col.lower() for col in data.columns]
    con.close()

    return data


def read_pyp_transition(path):
    con = sqlite3.connect(path)

    data = pd.read_sql_query("SELECT FEATURE_ID, TRANSITION_ID, 1 - PEP AS POSTERIOR FROM SCORE_TRANSITION INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID WHERE TRANSITION.TYPE!='' AND TRANSITION.DECOY=0;", con)

    data.columns = [col.lower() for col in data.columns]
    con.close()

    return data


def read_pyp_transition_peptidoforms(path):
    con = sqlite3.connect(path)

    data = pd.read_sql_query("SELECT TRANSITION_ID, PEPTIDE_ID FROM TRANSITION INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID WHERE TRANSITION.TYPE!='' AND TRANSITION.DECOY=0;", con)

    data.columns = [col.lower() for col in data.columns]
    con.close()

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
    
def prepare_transition_bm(data, ipf_h0):
    data_matrix = []
    peptidoforms = data['peptidoforms'].apply(pd.Series)[0].str.split("\|").apply(pd.Series, 1).stack().unique()

    for i,tr in data.iterrows():
        for pf in peptidoforms:
            tr_peptidoforms = tr['peptidoforms'].split("|")
            if pf in tr_peptidoforms:
                data_matrix.append({'prior': tr['precursor_peakgroup_pp'] / len(peptidoforms), 'evidence': tr['posterior'], 'hypothesis': pf, 'peptidoforms': tr['peptidoforms']})
            else:
                data_matrix.append({'prior': tr['precursor_peakgroup_pp'] / len(peptidoforms), 'evidence': 1 - tr['posterior'], 'hypothesis': pf, 'peptidoforms': tr['peptidoforms']})

        if ipf_h0:
            data_matrix.append({'prior': 1 - tr['precursor_peakgroup_pp'], 'evidence': 1 - tr['posterior'], 'hypothesis': 'h0', 'peptidoforms': tr['peptidoforms']})

    return pd.DataFrame(data_matrix)

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

    # get MS2-level peak group data
    if ipf_ms2_scoring:
        uis_ms2_pg_data = data[['feature_id','ms2_peakgroup_pp']].drop_duplicates()
    else:
        uis_ms2_pg_data = data['feature_id'].drop_duplicates()
        uis_ms2_pg_data['ms2_peakgroup_pp'] = np.nan

    # get MS2-level precursor data
    uis_ms2_precursor_data = data[data['ms2_precursor_pp'] >= 1-ipf_max_precursor_pep][['feature_id','ms2_precursor_pp']].drop_duplicates()

    # merge MS1- & MS2-level precursor and peak group data
    uis_precursor_data = uis_ms2_precursor_data.merge(uis_ms1_precursor_data, on=['feature_id'], how='outer').merge(uis_ms2_pg_data, on=['feature_id'], how='outer')

    # prepare precursor-level Bayesian model
    uis_precursor_data_bm = uis_precursor_data.groupby('feature_id').apply(prepare_precursor_bm).reset_index()

    # compute posterior precursor probability
    prec_pp_data = apply_bm(uis_precursor_data_bm)
    prec_pp_data = prec_pp_data.rename(columns=lambda x: x.replace('posterior', 'precursor_peakgroup_pp'))

    return prec_pp_data[prec_pp_data['hypothesis']][['feature_id','precursor_peakgroup_pp']]


def peptidoform_inference(data, ipf_max_transition_pep, ipf_h0):
    # compute transition posterior probabilities
    uis_transition_data_bm = data[(data['posterior'] >= ipf_max_transition_pep)].groupby('feature_id').apply(prepare_transition_bm, ipf_h0).reset_index()

    # compute posterior peptidoform probability
    pp_data = apply_bm(uis_transition_data_bm)
    pp_data = pp_data.rename(columns=lambda x: x.replace('posterior', 'pf_score'))

    # compute model-based FDR
    pp_data['qvalue'] = compute_model_fdr(pp_data['pf_score'].values)
    pp_data['pep'] = 1 - pp_data['pf_score']
    pp_data['PosteriorFullPeptideName'] = pp_data['hypothesis']

    # merge precursor-level data with UIS data
    result = pp_data.merge(data[['feature_id','precursor_peakgroup_pp']].drop_duplicates(), on=['feature_id'], how='inner')

    return result

def compute_model_fdr(data):
    fdrs = []
    for t in data:
        fdrs.append((1-data[data >= t]).sum() / len(data[data >= t]))
    return np.array(fdrs)


def infer_peptidoforms(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep):
    logging.info("start IPF (Inference of PeptidoForms)")

    uis_precursor_table = read_pyp_peakgroup_precursor(infile)

    # precursor-level inference
    logging.info("start precursor-level inference")
    uis_precursor_data = precursor_inference(uis_precursor_table, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep)
    uis_precursor_data = uis_precursor_data[(uis_precursor_data['precursor_peakgroup_pp'] >= 1-ipf_max_precursor_peakgroup_pep)]
    logging.info("end precursor-level inference")

    uis_transition_table = read_pyp_transition(infile)
    uis_transition_peptidoform_table = read_pyp_transition_peptidoforms(infile)

    # prepare data for peptidoform inference
    uis_transition_peptidoform_table['peptide_id'] = uis_transition_peptidoform_table['peptide_id'].apply(str)
    uis_transition_peptidoform_table = uis_transition_peptidoform_table.groupby('transition_id')['peptide_id'].agg(lambda x: '|'.join(x)).reset_index()
    uis_transition_peptidoform_table.columns = ['transition_id','peptidoforms']
    uis_transition_table['transition_id'] = uis_transition_table['transition_id'].astype(int)
    uis_transition_table = pd.merge(uis_transition_table, uis_transition_peptidoform_table, on='transition_id')
    uis_transition_table = pd.merge(uis_transition_table, uis_precursor_data, on='feature_id')

    # start posterior peptidoform-level inference
    logging.info("start peptidoform-level inference")
    uis_peptidoform_data = peptidoform_inference(uis_transition_table, ipf_max_transition_pep, ipf_h0)
    logging.info("end peptidoform-level inference")
    # end posterior peptidoform-level inference

    uis_peptidoform_data = uis_peptidoform_data[uis_peptidoform_data['hypothesis']!='h0'][['feature_id','hypothesis','precursor_peakgroup_pp','qvalue','pep']]
    uis_peptidoform_data.columns = ['FEATURE_ID','PEPTIDE_ID','PRECURSOR_PEAKGROUP_POSTERIOR','QVALUE','PEP']

    if infile != outfile:
        copyfile(infile, outfile)

    con = sqlite3.connect(outfile)

    uis_peptidoform_data.to_sql("SCORE_IPF", con, index=False, if_exists='replace')
    con.close()

    logging.info("end IPF")

