from __future__ import print_function

import os
import subprocess
import shutil
import sys

import pandas as pd
import sqlite3

import pytest

pd.options.display.width = 220
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


def _run_osw(regtest, temp_folder, transition_quantification=False, peptide=False, protein=False):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.osw")
    shutil.copy(data_path, temp_folder)
    # MS1-level
    cmdline = "pyprophet score --in=test_data.osw --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"

    # peptide-level
    cmdline += " peptide --pi0_lambda=0.001 0 0 --in=test_data.osw --context=run-specific"
    cmdline += " peptide --pi0_lambda=0.001 0 0 --in=test_data.osw --context=experiment-wide"
    cmdline += " peptide --pi0_lambda=0.001 0 0 --in=test_data.osw --context=global"

    # protein-level
    cmdline += " protein --pi0_lambda=0 0 0 --in=test_data.osw --context=run-specific"
    cmdline += " protein --pi0_lambda=0 0 0 --in=test_data.osw --context=experiment-wide"
    cmdline += " protein --pi0_lambda=0 0 0 --in=test_data.osw --context=global"

    # export
    cmdline += " export --in=test_data.osw --max_rs_peakgroup_pep=1"

    if not transition_quantification:
        cmdline += " --no-transition_quantification"

    if not peptide:
        cmdline += " --no-peptide"

    if not protein:
        cmdline += " --no-protein"

    stdout = _run_cmdline(cmdline)

    print(pd.read_csv("test_data.tsv", sep="\t", nrows=100),file=regtest)


def _run_ipf(regtest, temp_folder, transition_quantification=False, ipf=False):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.osw")
    shutil.copy(data_path, temp_folder)
    # MS1-level
    cmdline = "pyprophet score --in=test_data.osw --level=ms1 --test --pi0_lambda=0.1 0 0 --ss_iteration_fdr=0.02"

    # MS2-level
    cmdline += " score --in=test_data.osw --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"

    # transition-level
    cmdline += " score --in=test_data.osw --level=transition --test --pi0_lambda=0.1 0 0 --ss_iteration_fdr=0.02"

    # IPF
    cmdline += " ipf --in=test_data.osw"

    # export
    cmdline += " export --in=test_data.osw --no-peptide --no-protein --max_rs_peakgroup_pep=1"

    if not transition_quantification:
        cmdline += " --no-transition_quantification"

    if not ipf:
        cmdline += " --no-ipf"

    stdout = _run_cmdline(cmdline)

    print(pd.read_csv("test_data.tsv", sep="\t", nrows=100),file=regtest)


def test_osw_0(tmpdir, regtest):
    _run_osw(regtest, tmpdir.strpath, False, False, False)

def test_osw_1(tmpdir, regtest):
    _run_osw(regtest, tmpdir.strpath, True, False, False)

def test_osw_2(tmpdir, regtest):
    _run_osw(regtest, tmpdir.strpath, False, True, False)

def test_osw_3(tmpdir, regtest):
    _run_osw(regtest, tmpdir.strpath, False, False, True)

def test_ipf_0(tmpdir, regtest):
    _run_ipf(regtest, tmpdir.strpath, False, False)

def test_ipf_1(tmpdir, regtest):
    _run_ipf(regtest, tmpdir.strpath, True, False)

def test_ipf_2(tmpdir, regtest):
    _run_ipf(regtest, tmpdir.strpath, False, True)
