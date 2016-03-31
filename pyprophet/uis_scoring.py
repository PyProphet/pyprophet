# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import random

import pandas as pd
import numpy as np
import scipy as sp
from config import CONFIG

try:
    profile
except NameError:
    def profile(fun):
        return fun

from std_logger import logging

pd.set_option('chained_assignment',None)

def determine_output_dir_name(dirname, pathes):
    if dirname is None:
        dirnames = set(os.path.dirname(path) for path in pathes)
        # is always ok for not learning_mode, which includes that pathes has only one entry
        if len(dirnames) > 1:
            raise Exception("could not derive common directory name of input files, please use "
                            "--target.dir option")
        dirname = dirnames.pop()

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        logging.info("created folder %s" % dirname)
    return dirname

def prepare_precursor_bm(data):
    data_matrix = []

    for i,tr in data.iterrows():
        if tr['ms1_pg_score'] <= 1.0 and tr['ms1_pg_score'] >= 0:
            data_matrix.append({'prior': tr['ms2_pg_score'], 'evidence': tr['ms1_pg_score'], 'hypothesis': True})
            data_matrix.append({'prior': 1-tr['ms2_pg_score'], 'evidence': 1-tr['ms1_pg_score'], 'hypothesis': False})
        if tr['ms2_prec_score'] <= 1.0 and tr['ms2_prec_score'] >= 0:
            data_matrix.append({'prior': tr['ms2_pg_score'], 'evidence': tr['ms2_prec_score'], 'hypothesis': True})
            data_matrix.append({'prior': 1-tr['ms2_pg_score'], 'evidence': 1-tr['ms2_prec_score'], 'hypothesis': False})

    # There is no evidence for a precursor so we set the evidence to 0.
    if len(data_matrix) == 0:
            data_matrix.append({'prior': tr['ms2_pg_score'], 'evidence': 0, 'hypothesis': True})
            data_matrix.append({'prior': 1-tr['ms2_pg_score'], 'evidence': 1, 'hypothesis': False})

    return pd.DataFrame(data_matrix)
    
def prepare_transition_bm(data):
    data_matrix = []
    peptidoforms = data['peptidoforms'].apply(pd.Series)[0].str.split("\|").apply(pd.Series, 1).stack().unique()

    for i,tr in data.iterrows():
        for pf in peptidoforms:
            tr_peptidoforms = tr['peptidoforms'].split("|")
            if pf in tr_peptidoforms:
                data_matrix.append({'prior': tr['prec_pg_score'] / len(peptidoforms), 'evidence': tr['pg_score'], 'hypothesis': pf, 'peptidoforms': tr['peptidoforms']})
            else:
                data_matrix.append({'prior': tr['prec_pg_score'] / len(peptidoforms), 'evidence': 1 - tr['pg_score'], 'hypothesis': pf, 'peptidoforms': tr['peptidoforms']})

        if not CONFIG.get("uis_scoring.disable_h0"):
            data_matrix.append({'prior': 1 - tr['prec_pg_score'], 'evidence': 1 - tr['pg_score'], 'hypothesis': 'h0', 'peptidoforms': tr['peptidoforms']})

    return pd.DataFrame(data_matrix)

def apply_bm(data):
    # compute likelihood * prior per id & hypothesis
    # all priors are identical but pandas DF multiplication requires aggregation, so we use min()
    pp_data = (data.groupby(["id","hypothesis"])["evidence"].prod() * data.groupby(["id","hypothesis"])["prior"].min()).reset_index()
    pp_data.columns = ['id','hypothesis','likelihood_prior']

    # compute likelihood sum per id
    pp_data['likelihood_sum'] = pp_data.groupby("id")['likelihood_prior'].transform(np.sum)

    # compute posterior hypothesis probability
    pp_data['posterior'] = pp_data['likelihood_prior'] / pp_data['likelihood_sum']

    return pp_data.fillna(value = 0)

def precursor_inference(data):
    # get MS1-level precursor data
    if CONFIG.get("ms1_scoring.enable"):
        uis_ms1_precursor_data = data[data['ms1_pg_score'] >= CONFIG.get("uis_scoring.precursor_id_probability")][['id','ms1_pg_score']].drop_duplicates()
    else:
        uis_ms1_precursor_data = data[['id']].drop_duplicates()
        uis_ms1_precursor_data['ms1_pg_score'] = np.nan

    # get MS2-level peak group data
    if CONFIG.get("ms2_scoring.enable"):
        uis_ms2_pg_data = data[['id','ms2_pg_score']].drop_duplicates()
    else:
        uis_ms2_pg_data = data['id'].drop_duplicates()
        uis_ms2_pg_data['ms2_pg_score'] = np.nan

    # get MS2-level precursor data
    uis_ms2_precursor_data = data[data['annotation'] == "Precursor_i0"][(data['pg_score'] >= CONFIG.get("uis_scoring.precursor_id_probability"))][['id','pg_score']].drop_duplicates()
    uis_ms2_precursor_data = uis_ms2_precursor_data.rename(columns={'pg_score':'ms2_prec_score'})

    # merge MS1- & MS2-level precursor and peak group data
    uis_precursor_data = uis_ms2_precursor_data.merge(uis_ms1_precursor_data, on=['id'], how='outer').merge(uis_ms2_pg_data, on=['id'], how='outer')

    # prepare precursor-level Bayesian model
    uis_precursor_data_bm = uis_precursor_data.groupby("id").apply(prepare_precursor_bm).reset_index()

    # compute posterior precursor probability
    prec_pp_data = apply_bm(uis_precursor_data_bm)
    prec_pp_data = prec_pp_data.rename(columns=lambda x: x.replace('posterior', 'prec_pg_score'))

    # merge precursor-level data with UIS data
    result = data[data['annotation'] != "Precursor_i0"].merge(prec_pp_data[prec_pp_data['hypothesis']][['id','prec_pg_score']], on=['id'], how='outer')

    return result

def peptidoform_inference(data):
    # compute transition posterior probabilities
    uis_transition_data_bm = data[(data['pg_score'] >= CONFIG.get("uis_scoring.transition_id_probability"))].groupby("id").apply(prepare_transition_bm).reset_index()

    # compute posterior peptidoform probability
    pp_data = apply_bm(uis_transition_data_bm)
    pp_data = pp_data.rename(columns=lambda x: x.replace('posterior', 'pf_score'))

    # compute model-based FDR
    pp_data['pfqm_score'] = compute_model_fdr(pp_data['pf_score'].values)
    pp_data['PosteriorFullPeptideName'] = pp_data['hypothesis']

    # merge precursor-level data with UIS data
    result = pp_data.merge(data[['id','prec_pg_score']].drop_duplicates(), on=['id'], how='inner')

    return result

def compute_model_fdr(data):
    fdrs = []
    for t in data:
        fdrs.append((1-data[data >= t]).sum() / len(data[data >= t]))
    return np.array(fdrs)

def melt_uis_score_columns(table,cols):
    result = None
    for col in cols:
        molten = pd.melt(pd.concat([table['id'],table[col].str.split(";").apply(pd.Series)], axis=1),id_vars=['id'])
        molten = molten[pd.notnull(molten['value'])][molten['value'] != ""][['id','value']]
        molten.columns = ['id',col]
        if col.startswith("main_var_") or col.startswith("var_"):
            molten[col] = molten[col].convert_objects(convert_numeric=True)

        if result is None:
            result = molten
        else:
            result = pd.concat([result,molten[col]], axis=1, join_axes=[result.index])

    result = result.merge(table[['run_id','transition_group_id','id','ms1_pg_score','pg_score']], on=['id'], how='left')
    return result

def prepare_uis_tables(pathes):
    target_dir = determine_output_dir_name(CONFIG.get("target.dir"),pathes)

    new_pathes = []

    for path in pathes:
        table = pd.read_csv(path, CONFIG.get("delim.in"))

        table = table.rename(columns=lambda x: x.replace('uis_target_ind_log_intensity', 'uis_target_var_ind_log_intensity'))
        table = table.rename(columns=lambda x: x.replace('uis_target_ind_xcorr_coelution', 'uis_target_var_ind_xcorr_coelution'))
        table = table.rename(columns=lambda x: x.replace('uis_target_main_ind_xcorr_shape', 'uis_target_main_var_ind_xcorr_shape'))
        table = table.rename(columns=lambda x: x.replace('uis_target_ind_log_sn_score', 'uis_target_var_ind_log_sn_score'))
        table = table.rename(columns=lambda x: x.replace('uis_target_ind_massdev_score', 'uis_target_var_ind_massdev_score'))
        table = table.rename(columns=lambda x: x.replace('uis_target_ind_isotope_correlation', 'uis_target_var_ind_isotope_correlation'))
        table = table.rename(columns=lambda x: x.replace('uis_target_ind_isotope_overlap', 'uis_target_var_ind_isotope_overlap'))
        table = table.rename(columns=lambda x: x.replace('uis_decoy_ind_log_intensity', 'uis_decoy_var_ind_log_intensity'))
        table = table.rename(columns=lambda x: x.replace('uis_decoy_ind_xcorr_coelution', 'uis_decoy_var_ind_xcorr_coelution'))
        table = table.rename(columns=lambda x: x.replace('uis_decoy_main_ind_xcorr_shape', 'uis_decoy_main_var_ind_xcorr_shape'))
        table = table.rename(columns=lambda x: x.replace('uis_decoy_ind_log_sn_score', 'uis_decoy_var_ind_log_sn_score'))
        table = table.rename(columns=lambda x: x.replace('uis_decoy_ind_massdev_score', 'uis_decoy_var_ind_massdev_score'))
        table = table.rename(columns=lambda x: x.replace('uis_decoy_ind_isotope_correlation', 'uis_decoy_var_ind_isotope_correlation'))
        table = table.rename(columns=lambda x: x.replace('uis_decoy_ind_isotope_overlap', 'uis_decoy_var_ind_isotope_overlap'))
        
        # select uis scores
        uis_target_columns = [c for c in table.columns.values if c.startswith("uis_target_transition_names")] + [c for c in table.columns.values if c.startswith("uis_target_main_var_")] + [c for c in table.columns.values if c.startswith("uis_target_var_")]
        uis_decoy_columns = [c for c in table.columns.values if c.startswith("uis_decoy_transition_names")] + [c for c in table.columns.values if c.startswith("uis_decoy_main_var_")] + [c for c in table.columns.values if c.startswith("uis_decoy_var_")]

        # select target transition groups below or equal pg_score cutoff
        uis_targets = table[(table['pg_score'] >= CONFIG.get("uis_scoring.peakgroup_id_probability"))][table['decoy'] == 0][["run_id","transition_group_id","id","ms1_pg_score","pg_score"]+uis_target_columns]

        uis_targets = uis_targets.rename(columns=lambda x: x.replace('uis_target_', ''))

        # melt collapsed target transition-level scores to long list
        uis_targets_molten = melt_uis_score_columns(uis_targets,[col.replace('uis_target_','') for col in uis_target_columns])
        uis_targets_molten.ix[uis_targets_molten['transition_names'].str.find("Precursor_i0") > -1, 'annotation'] = "Precursor_i0"
        uis_targets_molten.ix[uis_targets_molten['transition_names'].str.find("Precursor_i0") == -1, 'annotation'] = uis_targets_molten[uis_targets_molten['transition_names'].str.find("Precursor_i0") == -1]['transition_names'].str.split("_").str[-1]
        uis_targets_molten['transition_group_id'] = uis_targets_molten.transition_group_id + "_tr_" + uis_targets_molten.transition_names + "_id_" + uis_targets_molten.id
        uis_targets_molten['decoy'] = 0
        uis_targets_molten = uis_targets_molten.rename(columns={'pg_score':'ms2_pg_score'})

        # remove all transitions that are part of a different isotopic pattern
        uis_targets_molten = uis_targets_molten[uis_targets_molten['var_ind_isotope_overlap'] < CONFIG.get("uis_scoring.isotope_overlap_threshold")]

        # select decoy transition groups below or equal q-value cutoff
        uis_decoys = table[(table['pg_score'] >= CONFIG.get("uis_scoring.peakgroup_id_probability"))][table['decoy'] == 0][["run_id","transition_group_id","id","ms1_pg_score","pg_score"]+uis_decoy_columns]

        uis_decoys = uis_decoys.rename(columns=lambda x: x.replace('uis_decoy_', ''))

        # melt collapsed decoy transition-level scores to long list
        uis_decoys_molten = melt_uis_score_columns(uis_decoys,[col.replace('uis_decoy_','') for col in uis_decoy_columns])
        uis_decoys_molten.ix[uis_decoys_molten['transition_names'].str.find("Precursor_i0") > -1, 'annotation'] = "Precursor_i0"
        uis_decoys_molten.ix[uis_decoys_molten['transition_names'].str.find("Precursor_i0") == -1, 'annotation'] = uis_decoys_molten[uis_decoys_molten['transition_names'].str.find("Precursor_i0") == -1]['transition_names'].str.split("_").str[-1]
        uis_decoys_molten['transition_group_id'] = uis_decoys_molten.transition_group_id + "_tr_" + uis_decoys_molten.transition_names + "_id_" + uis_decoys_molten.id
        uis_decoys_molten['transition_group_id'] = "DECOY_" + uis_decoys_molten['transition_group_id']
        uis_decoys_molten['id'] = "DECOY_" + uis_decoys_molten['id']
        uis_decoys_molten['decoy'] = 1
        uis_decoys_molten = uis_decoys_molten.rename(columns={'pg_score':'ms2_pg_score'})

        # remove all transitions that are part of a different isotopic pattern
        uis_decoys_molten = uis_decoys_molten[uis_decoys_molten['var_ind_isotope_overlap'] < CONFIG.get("uis_scoring.isotope_overlap_threshold")]

        # combine target and decoy transitions
        uis_table = pd.concat([uis_targets_molten,uis_decoys_molten]).sort_values(by='transition_group_id', ascending=[1])

        uis_table = uis_table.reset_index()

        new_path = target_dir + os.path.basename(path).split("_with_dscore.")[0] + "_uis." + path.split("_with_dscore.")[1]
        uis_table.to_csv(new_path , sep=CONFIG.get("delim.in"), index=False)
        new_pathes.append(new_path)

    return new_pathes

def postprocess_uis_tables(pathes, uis_res_pathes):
    target_dir = determine_output_dir_name(CONFIG.get("target.dir"),pathes)

    new_pathes = []

    uis_scored_table = pd.concat([pd.read_csv(uis_path, CONFIG.get("delim.in")) for uis_path in uis_res_pathes])

    # only apply inference to targets
    uis_scored_table = uis_scored_table[uis_scored_table['decoy'] == 0]
    uis_scored_table['peptidoforms'] = uis_scored_table['transition_group_id'].str.extract("\{(.*?)\}")

    # precursor-level inference
    logging.info("start precursor-level inference")
    uis_precursor_data = precursor_inference(uis_scored_table)
    uis_precursor_data = uis_precursor_data[(uis_precursor_data['prec_pg_score'] >= CONFIG.get("uis_scoring.prec_pg_id_probability"))]
    logging.info("end precursor-level inference")

    # start posterior peptidoform-level inference
    logging.info("start peptidoform-level inference")
    uis_peptidoform_data = peptidoform_inference(uis_precursor_data)
    logging.info("end peptidoform-level inference")
    # end posterior peptidoform-level inference

    if CONFIG.get("uis_scoring.expand_peptidoforms"):
        # expand peptidoforms for alignment using TRIC
        for path in pathes:
            table = pd.read_csv(path, CONFIG.get("delim.in"))
            
            expanded_table = table.merge(uis_peptidoform_data[['id','PosteriorFullPeptideName','prec_pg_score','pf_score','pfqm_score']], on=['id'], how='inner')
            expanded_table['transition_group_id'] = expanded_table['PosteriorFullPeptideName'] + '_' + expanded_table['transition_group_id']
            expanded_table['detection_m_score'] = expanded_table['m_score']
            expanded_table['m_score'] = expanded_table['pfqm_score']

            new_path = target_dir + os.path.basename(path).split(".")[0] + "_uis_expanded." + os.path.basename(path).split(".")[1]
            expanded_table.to_csv(new_path , sep=CONFIG.get("delim.in"), index=False)
            print "WRITTEN: ", new_path
            new_pathes.append(new_path)
    else:
        # collapse alternative peptidoforms with pf_scores
        uis_peptidoform_data['AlternativeFullPeptideName'] = uis_peptidoform_data['hypothesis'] + ":" + uis_peptidoform_data['pf_score'].map(str)
        uis_peptidoform_data['AlternativeFullPeptideName'] = uis_peptidoform_data.sort_values(by=['id','pf_score','hypothesis'],ascending=False).groupby(['id'])['AlternativeFullPeptideName'].transform(lambda x: ';'.join(x))

        # report only (multiple) best results
        uis_peptidoform_data['max_pf_score'] = uis_peptidoform_data.sort_values(by=['id','pf_score','hypothesis','AlternativeFullPeptideName'],ascending=False).groupby(['id'])['pf_score'].transform(max)

        # collapse (multiple) best FullPeptideName
        uis_peptidoform_data_collapsed = uis_peptidoform_data[uis_peptidoform_data['pf_score'] == uis_peptidoform_data['max_pf_score']].sort_values(by=['id','pf_score','hypothesis','AlternativeFullPeptideName'],ascending=False).groupby(['id']).agg({'PosteriorFullPeptideName': lambda x: ';'.join(x), 'prec_pg_score': lambda x: x.iloc[0], 'pf_score': lambda x: x.iloc[0], 'pfqm_score': lambda x: x.iloc[0], 'AlternativeFullPeptideName': lambda x: x.iloc[0]}).reset_index()

        for path in pathes:
                table = pd.read_csv(path, CONFIG.get("delim.in"))

                table = table.merge(uis_peptidoform_data_collapsed, on=['id'], how='inner')
                table.ix[pd.notnull(table.PosteriorFullPeptideName),'transition_group_id'] = table.ix[pd.notnull(table.PosteriorFullPeptideName)]['PosteriorFullPeptideName'] + "_" + table[pd.notnull(table.PosteriorFullPeptideName)]['transition_group_id']

                new_path = target_dir + os.path.basename(path).split(".")[0] + "_uis_collapsed." + os.path.basename(path).split(".")[1]
                table.to_csv(new_path , sep=CONFIG.get("delim.in"), index=False)
                print "WRITTEN: ", new_path
                new_pathes.append(new_path)

    logging.info("processing uis scores finished")
    return new_pathes

