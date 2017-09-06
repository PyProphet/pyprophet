import os
import subprocess
import shutil
import sys

import pandas as pd

import pytest


d = pd.options.display
d.width = 220
d.precision = 6

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _record(stdout, regtest):
    for line in stdout.split("\n"):
        if line.startswith("INFO"):
            # remove data which is not constant over runs as time or process id:
            line = line.split(":", 4)[-1]
        if "TIME" in line:
            continue
        print >> regtest, line


def _expected_output_files():
    return ["test_data_ms2_scored.tsv",
            "test_data_ms2_report.pdf",
            "test_data_ms2_weights.csv",
            "test_data_ms2_summary_stat.csv",
            "test_data_ms2_full_stat.csv",
            ]


def _remove_output_files(tmpdir, names=None):
    if names is None:
        names = _expected_output_files()
    for name in names:
        full_path = os.path.join(tmpdir.strpath, name)
        os.remove(full_path)


def _dump_output_files(tmpdir, regtest, names=None):
    try:
        tmpdir = tmpdir.strpath
    except AttributeError:
        pass
    if names is None:
        names = _expected_output_files()
    for name in names:
        full_path = os.path.join(tmpdir, name)
        if not name.endswith(".pdf"):
            _dump(full_path, regtest)


def _dump(full_path, regtest):
    head = open(full_path).readline()
    if "\t" in head:
        df = pd.read_csv(full_path, sep="\t", header=None)
    else:
        df = pd.read_csv(full_path, sep=" ", header=None)

    # lines = open(full_path, "r").readlines()
    # f = os.path.basename(full_path)
    if len(df) > 20:
        print >> regtest, df[:10]
        print >> regtest, "..."
        print >> regtest, df[-10:]
    else:
        print >> regtest, df


def _run_cmdline(cmdline):
    stdout = cmdline + "\n"
    try:
        stdout += subprocess.check_output(cmdline, shell=True,
                                          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        print >> sys.stderr, e.output
        raise
    return stdout


def _run_pyprophet_to_learn_model(regtest, temp_folder, dump_result_files=False, parametric=False, pfdr=False, pi0_lambda=False):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.txt")
    shutil.copy(data_path, temp_folder)
    cmdline = "pyprophet score --in=test_data.txt --test"
    if parametric:
        cmdline += " --parametric"
    if pfdr:
        cmdline += " --pfdr"
    if pi0_lambda is not False:
        cmdline += " --pi0_lambda=" + pi0_lambda
    stdout = _run_cmdline(cmdline)

    for f in _expected_output_files():
        full_path = os.path.join(temp_folder, f)
        assert os.path.exists(full_path)

    if not dump_result_files:
        return stdout

    _dump_output_files(temp_folder, regtest)

    return stdout


def test_tsv_0(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True)
    _record(stdout, regtest)

def test_tsv_1(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, True)
    _record(stdout, regtest)

def test_tsv_2(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, False, True)
    _record(stdout, regtest)

def test_tsv_3(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, False, False, "0.3 0.55 0.05")
    _record(stdout, regtest)

def test_tsv_apply_weights(tmpdir, regtest):

    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True)

    _record(stdout, regtest)

    full_path = os.path.join(tmpdir.strpath, "test_data_ms2_scored.tsv")
    _dump(full_path, regtest)

    # collect q-value stats
    m_score_stat, = [l for l in stdout.split("\n") if "mean q_value" in l]

    # split away log time etc:
    __, __, interesting_m_score_output = m_score_stat.partition("mean q_value")

    # collect s value stats
    s_value_stat, = [l for l in stdout.split("\n") if "mean s_value" in l]

    # split away log time etc:
    __, __, interesting_s_value_output = s_value_stat.partition("mean s_value")

    stdout = _run_cmdline("pyprophet score --in=test_data.txt --apply_weights=test_data_ms2_weights.csv "
                          "--test")

    _record(stdout, regtest)

    # collect m score stats
    m_score_stat, = [l for l in stdout.split("\n") if "mean q_value" in l]

    # split away log time etc:
    __, __, interesting_m_score_output2 = m_score_stat.partition("mean q_value")

    # collect s value stats
    s_value_stat, = [l for l in stdout.split("\n") if "mean s_value" in l]

    # split away log time etc:
    __, __, interesting_s_value_output2 = s_value_stat.partition("mean s_value")

    # check if output is consistent over runs:
    assert interesting_m_score_output == interesting_m_score_output2
    assert interesting_s_value_output == interesting_s_value_output2


def test_not_unique_tg_id_blocks(tmpdir):

    os.chdir(tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_invalid_data.txt")
    shutil.copy(data_path, tmpdir.strpath)
    cmdline = "pyprophet score --in=test_invalid_data.txt --test"

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.check_output(cmdline, shell=True, stderr=subprocess.STDOUT)

    e = exc_info.value
    assert "Error: group_id values do not form unique blocks in input file(s)." in e.output
