from __future__ import print_function

import os
import subprocess
import shutil
import sys

import pandas as pd
import sqlite3

import pytest

from pyprophet.ipf import read_pyp_peakgroup_precursor

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
pd.options.display.width = 1000
pd.options.display.precision = 4

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


def _run_pyprophet_tsv_to_learn_model(regtest, temp_folder, dump_result_files=False, parametric=False, pfdr=False, pi0_lambda=False):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.txt")
    shutil.copy(data_path, temp_folder)
    cmdline = "pyprophet score --pi0_method=smoother --pi0_lambda 0.4 0 0 --in=test_data.txt --test --ss_iteration_fdr=0.02"
    if parametric:
        cmdline += " --parametric"
    if pfdr:
        cmdline += " --pfdr"
    if pi0_lambda is not False:
        cmdline += " --pi0_lambda=" + pi0_lambda
    stdout = _run_cmdline(cmdline)

    print(pd.read_csv("test_data_summary_stat.csv", sep=",", nrows=100),file=regtest)
    print(pd.read_csv("test_data_full_stat.csv", sep=",", nrows=100),file=regtest)
    print(pd.read_csv("test_data_scored.tsv", sep="\t", nrows=100),file=regtest)
    print(pd.read_csv("test_data_weights.csv", sep=",", nrows=100),file=regtest)


def _run_pyprophet_osw_to_learn_model(regtest, temp_folder, dump_result_files=False, parametric=False, pfdr=False, pi0_lambda=False):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.osw")
    shutil.copy(data_path, temp_folder)
    # MS1-level
    cmdline = "pyprophet score --in=test_data.osw --level=ms1 --test --ss_iteration_fdr=0.02"
    if parametric:
        cmdline += " --parametric"
    if pfdr:
        cmdline += " --pfdr"
    if pi0_lambda is not False:
        cmdline += " --pi0_lambda=" + pi0_lambda

    # MS2-level
    cmdline += " score --in=test_data.osw --level=ms2 --test --ss_iteration_fdr=0.02"
    if parametric:
        cmdline += " --parametric"
    if pfdr:
        cmdline += " --pfdr"
    if pi0_lambda is not False:
        cmdline += " --pi0_lambda=" + pi0_lambda

    # transition-level
    cmdline += " score --in=test_data.osw --level=transition --test --ss_iteration_fdr=0.02"
    if parametric:
        cmdline += " --parametric"
    if pfdr:
        cmdline += " --pfdr"
    if pi0_lambda is not False:
        cmdline += " --pi0_lambda=" + pi0_lambda

    stdout = _run_cmdline(cmdline)

    table = read_pyp_peakgroup_precursor("test_data.osw", 1.0, True, True)

    print(table.head(100),file=regtest)

def test_tsv_0(tmpdir, regtest):
    _run_pyprophet_tsv_to_learn_model(regtest, tmpdir.strpath, True)

def test_tsv_1(tmpdir, regtest):
    _run_pyprophet_tsv_to_learn_model(regtest, tmpdir.strpath, True, True)

def test_tsv_2(tmpdir, regtest):
    _run_pyprophet_tsv_to_learn_model(regtest, tmpdir.strpath, True, False, True)

def test_tsv_3(tmpdir, regtest):
    _run_pyprophet_tsv_to_learn_model(regtest, tmpdir.strpath, True, False, False, "0.3 0.55 0.05")

def test_tsv_apply_weights(tmpdir, regtest):

    _run_pyprophet_tsv_to_learn_model(regtest, tmpdir.strpath, True)

    _run_cmdline("pyprophet score --pi0_method=smoother --pi0_lambda 0.4 0 0 --in=test_data.txt --apply_weights=test_data_weights.csv "
                          "--test --ss_iteration_fdr=0.02")

def test_osw_0(tmpdir, regtest):
    _run_pyprophet_osw_to_learn_model(regtest, tmpdir.strpath, True, pi0_lambda="0 0 0")

def test_osw_1(tmpdir, regtest):
    _run_pyprophet_osw_to_learn_model(regtest, tmpdir.strpath, True, True, pi0_lambda="0 0 0")

def test_osw_2(tmpdir, regtest):
    _run_pyprophet_osw_to_learn_model(regtest, tmpdir.strpath, True, False, True, pi0_lambda="0 0 0")

def test_not_unique_tg_id_blocks(tmpdir):

    os.chdir(tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_invalid_data.txt")
    shutil.copy(data_path, tmpdir.strpath)
    cmdline = "pyprophet score --pi0_method=smoother --pi0_lambda 0.4 0 0 --in=test_invalid_data.txt --test --ss_iteration_fdr=0.02"

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.check_output(cmdline, shell=True, stderr=subprocess.STDOUT)

    e = exc_info.value
    assert "Error: group_id values do not form unique blocks in input file(s)." in str(e.output)
