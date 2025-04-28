from __future__ import print_function

import os
import subprocess
import shutil
import sys

import pandas as pd
import sqlite3

import pytest


pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def _run_cmdline(cmdline):
    stdout = cmdline + "\n"
    try:
        stdout += str(subprocess.check_output(cmdline, shell=True,
                                          stderr=subprocess.STDOUT))
    except subprocess.CalledProcessError as error:
        print(error, end="", file=sys.stderr)
        raise
    return stdout


def _run_export_parquet_single_run(temp_folder, transitionLevel=False, pd_testing_kwargs=dict(check_dtype=False, check_names=False), onlyFeatures=False, noDecoys=False):
    os.chdir(temp_folder)
    DATA_NAME="dummyOSWScoredData.osw"
    data_path = os.path.join(DATA_FOLDER, DATA_NAME)
    conn = sqlite3.connect(DATA_NAME)
    shutil.copy(data_path, temp_folder)
    
    cmdline = "pyprophet export-parquet --in={}".format(DATA_NAME)

    # if testing transition level add --transitionLevel flag
    if transitionLevel:
        cmdline += " --transitionLevel"
    if onlyFeatures:
        cmdline += " --onlyFeatures"
    if noDecoys:
        cmdline += " --noDecoys"

    stdout = _run_cmdline(cmdline)

    ### This file was configured in a way where the following tests should work
    parquet = pd.read_parquet("dummyOSWScoredData.parquet") ## automatically with parquet ending of input file name

    ### CHECK LENGTHS ###
    if transitionLevel:
        if onlyFeatures: # length of FEATURE_TRANSITION table
            if not noDecoys:
                expectedLength = len(pd.read_sql("select * from feature_transition", conn)) 
            else:
                expectedLength = len(pd.read_sql("select * from feature_transition inner join transition on transition.id = feature_transition.transition_id where DECOY == 0", conn))
        else:
            if not noDecoys:
                featureTransition = pd.read_sql("select * from feature_transition", conn)
                precursorTransition = pd.read_sql("select * from transition_precursor_mapping", conn)
            else:
                featureTransition = pd.read_sql("select * from feature_transition inner join transition on transition.id = feature_transition.transition_id where DECOY == 0", conn)
                precursorTransition = pd.read_sql("select * from transition_precursor_mapping inner join transition on transition.id = transition_precursor_mapping.transition_id where DECOY=0", conn)

            featureTable = pd.read_sql("select * from feature", conn)

            numTransNoFeature = len(precursorTransition[~precursorTransition['PRECURSOR_ID'].isin(featureTable['PRECURSOR_ID'])])

            expectedLength = numTransNoFeature + len(featureTransition)

    else:
        if onlyFeatures: # expected length, length of feature table
            if noDecoys:
                expectedLength = len(pd.read_sql("select * from feature inner join precursor on feature.precursor_id = precursor.id where decoy = 0", conn)) 
            else:
                expectedLength = len(pd.read_sql("select * from feature inner join precursor on precursor.id = feature.precursor_id", conn)) 
        else:
            # Expected length is number of features + number of precursors with no feature
            if noDecoys:
                featureTable = pd.read_sql("select * from feature inner join precursor on feature.precursor_id = precursor.id where decoy = 0", conn)
            else:
                featureTable = pd.read_sql("select * from feature", conn)

            if noDecoys:
                precTable = pd.read_sql("select * from precursor where decoy = 0", conn)
            else:
                precTable = pd.read_sql("select * from precursor", conn)

            numPrecsNoFeature = len(precTable[~precTable['ID'].isin(featureTable['PRECURSOR_ID'])])
            expectedLength = numPrecsNoFeature + len(featureTable)

    assert(expectedLength == len(parquet))


    ########### FEATURE LEVEL VALUE TESTS ########
    # Tests that columns are equal across different sqlite3 tables to ensure joins occured correctly

    # since cannot compare NAN drop rows which contain an NAN
    na_columns = ['PRECURSOR.LIBRARY_INTENSITY'] # this is a list of columns which expect to be NAN
    parquet = parquet.drop(columns=na_columns).dropna()

    assert(len(parquet) > 0) # assert that did not just drop everything (means that missed an na column)

    if transitionLevel:
        ## check features and transitions joined properly for those all cases (including those with no features
        ## Way library was created precursor and transition m/z both are in the same 100s (e.g. if precursor m/z is 700 transition mz can be 701) 
        pd.testing.assert_series_equal(parquet['PRECURSOR.PRECURSOR_MZ'] // 100, parquet['TRANSITION.PRODUCT_MZ'] // 100, **pd_testing_kwargs)

    ### Note: Current tests assume no na
    parquet = parquet.dropna()
    pseudo_feature_id = (parquet['FEATURE_ID'].astype(str).str.slice(start=0, stop=1)).astype(int)
    pd.testing.assert_series_equal(parquet['FEATURE_MS1.APEX_INTENSITY'], parquet['PRECURSOR_ID'], **pd_testing_kwargs)
    pd.testing.assert_series_equal(parquet['FEATURE_MS2.APEX_INTENSITY'],  parquet['PRECURSOR_ID'], **pd_testing_kwargs)

    pd.testing.assert_series_equal(parquet['FEATURE_MS1.EXP_IM'], parquet['FEATURE_MS2.EXP_IM'], **pd_testing_kwargs)
    pd.testing.assert_series_equal(parquet['FEATURE_MS2.DELTA_IM'],  parquet['FEATURE_MS1.DELTA_IM'], **pd_testing_kwargs)

    pd.testing.assert_series_equal(parquet['SCORE_MS2.SCORE'],  (parquet['PRECURSOR_ID'] + 1) * parquet['FEATURE.EXP_RT'].astype(int) * pseudo_feature_id, **pd_testing_kwargs)
    pd.testing.assert_series_equal(parquet['SCORE_PEPTIDE.SCORE_GLOBAL'],  parquet['PEPTIDE_ID'], **pd_testing_kwargs)
    pd.testing.assert_series_equal(parquet['SCORE_PROTEIN.SCORE_GLOBAL'],  parquet['PROTEIN_ID'], **pd_testing_kwargs)

    # check is/no decoys
    if noDecoys:
         assert(parquet[parquet['DECOY'] == 1].shape[0] == 0)
   


    ############### TRANSTION LEVEL TESTS ################
    if transitionLevel:
        pd.testing.assert_series_equal(parquet['FEATURE_TRANSITION.AREA_INTENSITY'], parquet['TRANSITION.PRODUCT_MZ'] * pseudo_feature_id, **pd_testing_kwargs)
	
 

def _run_export_parquet_scoring_format(regtest, temp_folder, compression="zstd", compression_level=11):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.osw")
    shutil.copy(data_path, temp_folder)

    # MS1-level
    cmdline = "pyprophet export-parquet --in=test_data.osw --out=test_data.parquet --scoring_format --compression {} --compression_level {}".format(compression, compression_level)
    
    stdout = _run_cmdline(cmdline)
    
    print(pd.read_parquet("test_data.parquet", engine="pyarrow").head(100).sort_index(axis=1),file=regtest)
 
 
def test_export_parquet_single_run(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=False)

	
def test_export_parquet_single_run_transitionLevel(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=True)


def test_export_parquet_single_run_onlyFeatures(tmpdir):
	_run_export_parquet_single_run(tmpdir, onlyFeatures=True)


def test_export_parquet_single_run_transitionLevel_onlyFeatures(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=True, onlyFeatures=True)
     
def test_export_parquet_single_run_noDecoys(tmpdir):
    _run_export_parquet_single_run(tmpdir, noDecoys=True)

def test_export_parquet_single_run_transitionLevel_noDecoys(tmpdir):
    _run_export_parquet_single_run(tmpdir, transitionLevel=True, noDecoys=True)
    
def test_osw_to_parquet_scoring_format_0(tmpdir, regtest):
    _run_export_parquet_scoring_format(regtest, tmpdir.strpath)
    
def test_osw_to_parquet_scoring_format_1(tmpdir, regtest):
    _run_export_parquet_scoring_format(regtest, tmpdir.strpath, 'snappy', 0)
    
def test_osw_to_parquet_scoring_format_2(tmpdir, regtest):
    _run_export_parquet_scoring_format(regtest, tmpdir.strpath, 'brotli', 5)
    
def test_osw_to_parquet_scoring_format_3(tmpdir, regtest):
    _run_export_parquet_scoring_format(regtest, tmpdir.strpath, 'gzip', 5)