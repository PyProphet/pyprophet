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


def _run_export_parquet_single_run(temp_folder, transitionLevel=False, threads=1, chunksize=1000, pd_testing_kwargs=dict(check_dtype=False, check_names=False), onlyFeatures=False):
    os.chdir(temp_folder)
    DATA_NAME="dummyOSWScoredData.osw"
    data_path = os.path.join(DATA_FOLDER, DATA_NAME)
    conn = sqlite3.connect(DATA_NAME)
    shutil.copy(data_path, temp_folder)

    cmdline = "pyprophet export-parquet --in={} --threads={} --chunksize={}".format(DATA_NAME, threads, chunksize)

    # if testing transition level add --transitionLevel flag
    if transitionLevel:
        cmdline += " --transitionLevel"
    if onlyFeatures:
        cmdline += " --onlyFeatures"

    stdout = _run_cmdline(cmdline)

    ### This file was configured in a way where the following tests should work
    parquet = pd.read_parquet("dummyOSWScoredData.parquet") ## automatically with parquet ending of input file name

    if transitionLevel:
        if onlyFeatures: # length of FEATURE_TRANSITION table
            expectedLength = len(pd.read_sql("select * from feature_transition", conn)) 
        else:
            featureTransition = pd.read_sql("select * from feature_transition", conn)
            precursorTransition = pd.read_sql("select * from transition_precursor_mapping", conn)
            featureTable = pd.read_sql("select * from feature", conn)

            numTransNoFeature = len(precursorTransition[~precursorTransition['PRECURSOR_ID'].isin(featureTable['PRECURSOR_ID'])])

            expectedLength = numTransNoFeature + len(featureTransition)

        assert(expectedLength == len(parquet))
    else:
        if onlyFeatures: # expected length, length of feature table
            expectedLength = len(pd.read_sql("select * from feature", conn)) 
        else:
            # Expected length is number of features + number of precursors with no feature
            featureTable = pd.read_sql("select * from feature", conn)
            precTable = pd.read_sql("select * from precursor", conn)

            numPrecsNoFeature = len(precTable[~precTable['ID'].isin(featureTable['PRECURSOR_ID'])])
            expectedLength = numPrecsNoFeature + len(featureTable)

        assert(expectedLength == len(parquet))

    ########### FEATURE LEVEL TESTS ########
    # Tests that columns are equal across different sqlite3 tables to ensure joins occured correctly

    if transitionLevel:
        ## check features and transitions joined properly for those all cases (including those with no features
        ## Way library was created precursor and transition m/z both are in the same 100s (e.g. if precursor m/z is 700 transition mz can be 701) 
        pd.testing.assert_series_equal(parquet['PRECURSOR.PRECURSOR_MZ'] // 100, parquet['TRANSITION.PRODUCT_MZ'] // 100, **pd_testing_kwargs)

    ### Note: Current tests assume no na
    parquet = parquet.dropna()
    proxy_feature_id = parquet['FEATURE_ID'].astype(str).apply(lambda x: x[0]).astype(int) # since id is complicated, dummy values created using a proxy id which is the first digit of the actual id
    pd.testing.assert_series_equal(parquet['FEATURE_MS1.APEX_INTENSITY'], parquet['PRECURSOR_ID'], **pd_testing_kwargs)
    pd.testing.assert_series_equal(parquet['FEATURE_MS2.APEX_INTENSITY'],  parquet['PRECURSOR_ID'], **pd_testing_kwargs)

    pd.testing.assert_series_equal(parquet['FEATURE_MS1.EXP_IM'], parquet['FEATURE_MS2.EXP_IM'], **pd_testing_kwargs)
    pd.testing.assert_series_equal(parquet['FEATURE_MS2.DELTA_IM'],  parquet['FEATURE_MS1.DELTA_IM'], **pd_testing_kwargs)

    pd.testing.assert_series_equal(parquet['SCORE_MS2.SCORE'],  (parquet['PRECURSOR_ID'] + 1) * parquet['FEATURE.EXP_RT'].astype(int) * (proxy_feature_id), **pd_testing_kwargs)
    print(parquet.columns)
    pd.testing.assert_series_equal(parquet['SCORE_PEPTIDE.SCORE_GLOBAL'],  parquet['PEPTIDE_ID'], **pd_testing_kwargs)
    pd.testing.assert_series_equal(parquet['SCORE_PROTEIN.SCORE_GLOBAL'],  parquet['PROTEIN_ID'], **pd_testing_kwargs)

    ############### TRANSTION LEVEL TESTS ################
    if transitionLevel:
        pd.testing.assert_series_equal(parquet['FEATURE_TRANSITION.AREA_INTENSITY'], parquet['TRANSITION.PRODUCT_MZ'] * (proxy_feature_id), **pd_testing_kwargs)
	
def test_export_parquet_single_run(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=False)

	
def test_export_parquet_single_run_transitionLevel(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=True)


def test_export_parquet_single_run_onlyFeatures(tmpdir):
	_run_export_parquet_single_run(tmpdir, onlyFeatures=True)


def test_export_parquet_single_run_transitionLevel_onlyFeatures(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=True, onlyFeatures=True)


def test_multithread_export_parquet_single_run(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=False, threads=2, chunksize=2)

def test_multithread_export_parquet_single_run_transitionLevel(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=True, threads=2, chunksize=2)


def test_multithread_export_parquet_single_run_onlyFeatures(tmpdir):
	_run_export_parquet_single_run(tmpdir, onlyFeatures=True, threads=2, chunksize=4)


def test_multithread_export_parquet_single_run_transitionLevel_onlyFeatures(tmpdir):
	_run_export_parquet_single_run(tmpdir, transitionLevel=True, onlyFeatures=True, threads=2, chunksize=4)
