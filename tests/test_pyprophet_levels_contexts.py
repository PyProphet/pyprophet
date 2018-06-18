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


def _run_peptide(regtest, temp_folder):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.osw")
    shutil.copy(data_path, temp_folder)
    # MS1-level
    cmdline = "pyprophet score --in=test_data.osw --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"

    # peptide-level
    cmdline += " peptide --pi0_lambda=0.001 0 0 --in=test_data.osw --context=run-specific"
    cmdline += " peptide --pi0_lambda=0.001 0 0 --in=test_data.osw --context=experiment-wide"
    cmdline += " peptide --pi0_lambda=0.001 0 0 --in=test_data.osw --context=global"

    stdout = _run_cmdline(cmdline)

    con = sqlite3.connect("test_data.osw")

    # validate transition-level data
    transition_data = pd.read_sql_query("SELECT * FROM SCORE_PEPTIDE LIMIT 100;", con)
    transition_data.columns = [col.lower() for col in transition_data.columns]

    print(transition_data.sort_index(axis=1).head(100),file=regtest)

    con.close()


def _run_protein(regtest, temp_folder):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.osw")
    shutil.copy(data_path, temp_folder)
    # MS1-level
    cmdline = "pyprophet score --in=test_data.osw --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"

    # protein-level
    cmdline += " protein --pi0_lambda=0 0 0 --in=test_data.osw --context=run-specific"
    cmdline += " protein --pi0_lambda=0 0 0 --in=test_data.osw --context=experiment-wide"
    cmdline += " protein --pi0_lambda=0 0 0 --in=test_data.osw --context=global"

    stdout = _run_cmdline(cmdline)

    con = sqlite3.connect("test_data.osw")

    # validate transition-level data
    transition_data = pd.read_sql_query("SELECT * FROM SCORE_PROTEIN LIMIT 100;", con)
    transition_data.columns = [col.lower() for col in transition_data.columns]

    print(transition_data.sort_index(axis=1).head(100),file=regtest)

    con.close()


def test_levels_contexts_0(tmpdir, regtest):
    _run_peptide(regtest, tmpdir.strpath)

def test_ipf_1(tmpdir, regtest):
    _run_protein(regtest, tmpdir.strpath)
