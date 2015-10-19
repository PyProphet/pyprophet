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

def prepare_ms1_tables(pathes):
    target_dir = determine_output_dir_name(CONFIG.get("target.dir"),pathes)

    new_pathes = []
    for path in pathes:
        table = pd.read_csv(path, CONFIG.get("delim.in"))

        table = table.rename(columns=lambda x: x.replace('var_ms1_', 'ms1_'))
        table = table.rename(columns=lambda x: x.replace('ms1_', 'var_ms1_'))

        ms1_var_columns = [c for c in table.columns.values if c.startswith("var_ms1_")]
        ms1_table = table[["run_id","transition_group_id","id","decoy"]+ms1_var_columns].reset_index()
        ms1_table = ms1_table.rename(columns=lambda x: x.replace('var_ms1_xcorr_shape', 'main_var_ms1_xcorr_shape'))

        new_path = target_dir + os.path.basename(path).split(".")[0] + "_ms1." + os.path.basename(path).split(".")[1]
        ms1_table.to_csv(new_path , sep=CONFIG.get("delim.in"), index=False)
        new_pathes.append(new_path)
    return new_pathes

def postprocess_ms1_tables(pathes, ms1_res_pathes):
    target_dir = determine_output_dir_name(CONFIG.get("target.dir"),pathes)

    new_pathes = []

    ms1_df = pd.concat([pd.read_csv(ms1_path, CONFIG.get("delim.in")) for ms1_path in ms1_res_pathes])

    if not CONFIG.get("ms2_scoring.disable_ms1_propagation"):
        ms1_ids = set(ms1_df[ms1_df['pg_score'] >= CONFIG.get("ms2_scoring.precursor_id_probability")]['transition_group_id'].str.replace("DECOY_","").tolist())

        ms1_merge_ids = []
        for entry in ms1_ids:
            ms1_merge_ids.append(entry)
            ms1_merge_ids.append("DECOY_" + entry)

        ms1_merge_table = ms1_df[ms1_df['transition_group_id'].isin(ms1_merge_ids)][['id','d_score','m_score','peak_group_rank','pg_score']]
    else:
        ms1_merge_table = ms1_df[ms1_df['m_score'] >= CONFIG.get("ms2_scoring.precursor_id_probability")][['id','d_score','m_score','peak_group_rank','pg_score']]

    ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('d_score', 'ms1_d_score'))
    ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('m_score', 'ms1_m_score'))
    ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('peak_group_rank', 'ms1_peak_group_rank'))
    ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('pg_score', 'ms1_pg_score'))

    if 'pyprophet_m_score' in ms1_merge_table.columns:
        ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('pyprophet_m_score', 'ms1_pyprophet_m_score'))

    if 'qvality_m_score' in ms1_merge_table.columns:
        ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('qvality_m_score', 'ms1_qvality_m_score'))

    if 'pg_score' in ms1_merge_table.columns:
        ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('pg_score', 'ms1_pg_score'))

    if 'h_score' in ms1_merge_table.columns:
        ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('h_score', 'ms1_h_score'))

    if 'h0_score' in ms1_merge_table.columns:
        ms1_merge_table = ms1_merge_table.rename(columns=lambda x: x.replace('h0_score', 'ms1_h0_score'))

    for path in pathes:
        table = pd.read_csv(path, CONFIG.get("delim.in"))
        table = table.merge(ms1_merge_table, on=['id'], how='inner')
        new_path = target_dir + os.path.basename(path).split(".")[0] + "_ms1_ms2." + os.path.basename(path).split(".")[1]
        table.to_csv(new_path , sep=CONFIG.get("delim.in"), index=False)
        new_pathes.append(new_path)

    logging.info("processing ms1 scores finished")
    return new_pathes

