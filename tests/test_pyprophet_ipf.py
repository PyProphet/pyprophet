from __future__ import print_function

import os
import subprocess
import shutil
import sys

import pandas as pd
import sqlite3

import pytest

from pyprophet.ipf import read_pyp_peakgroup_precursor

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


def _run_ipf(regtest, temp_folder, dump_result_files=False, ipf_ms1_scoring=True, ipf_ms2_scoring=True, ipf_h0=True):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.osw")
    shutil.copy(data_path, temp_folder)
    # MS1-level
    cmdline = "pyprophet score --in=test_data.osw --level=ms1 --test --pi0_lambda=0.1 0 0 --ss_iteration_fdr=0.02"

    # MS2-level
    cmdline += " score --in=test_data.osw --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"

    # transition-level
    cmdline += " score --in=test_data.osw --level=transition --test --pi0_lambda=0.1 0 0 --ss_iteration_fdr=0.02"

    # transition-level
    cmdline += " ipf --in=test_data.osw"
    if ipf_ms1_scoring:
        cmdline += " --ipf_ms1_scoring"
    else:
        cmdline += " --no-ipf_ms1_scoring"
    if ipf_ms2_scoring:
        cmdline += " --ipf_ms2_scoring"
    else:
        cmdline += " --no-ipf_ms2_scoring"
    if ipf_h0:
        cmdline += " --ipf_h0"
    else:
        cmdline += " --no-ipf_h0"

    stdout = _run_cmdline(cmdline)

    con = sqlite3.connect("test_data.osw")

    # validate precursor-level data
    precursor_data = read_pyp_peakgroup_precursor("test_data.osw", 1.0, True, True)
    print(precursor_data.sort_index(axis=1).head(100),file=regtest)

    # validate transition-level data
    transition_data = pd.read_sql_query("SELECT * FROM SCORE_TRANSITION LIMIT 100;", con)
    transition_data.columns = [col.lower() for col in transition_data.columns]

    print(transition_data.sort_index(axis=1).head(100),file=regtest)

    # validate IPF-level data
    ipf_data = pd.read_sql_query("SELECT * FROM SCORE_IPF LIMIT 100;", con)
    ipf_data.columns = [col.lower() for col in ipf_data.columns]

    print(ipf_data.sort_index(axis=1).head(100),file=regtest)

    con.close()


def test_ipf_0(tmpdir, regtest):
    _run_ipf(regtest, tmpdir.strpath, True, True, True, True)

def test_ipf_1(tmpdir, regtest):
    _run_ipf(regtest, tmpdir.strpath, True, False, True, True)

def test_ipf_2(tmpdir, regtest):
    _run_ipf(regtest, tmpdir.strpath, True, True, False, True)

def test_ipf_3(tmpdir, regtest):
    _run_ipf(regtest, tmpdir.strpath, True, False, False, True)

def test_ipf_4(tmpdir, regtest):
    _run_ipf(regtest, tmpdir.strpath, True, True, True, False)
