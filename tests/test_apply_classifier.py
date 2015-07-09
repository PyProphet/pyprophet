import os
import subprocess
import shutil

import pytest

__here__ = os.path.dirname(os.path.abspath(__file__))


def _record(stdout, regtest):
    for line in stdout.split("\n"):
        if line.startswith("INFO"):
            # remove data which is not constant over runs as time or process id:
            line = line.split(":", 4)[-1]
        if "NEEDED" in line:
            continue
        if "time needed" in line:
            continue
        print >> regtest, line


def _setup_first_run(tmpdir):
    os.chdir(tmpdir.strpath)
    data_path = os.path.join(__here__, "test_data.txt")
    shutil.copy(data_path, tmpdir.strpath)
    stdout = subprocess.check_output("pyprophet test_data.txt --is_test", shell=True,
                                     stderr=subprocess.STDOUT)
    return stdout


def test_apply_classifier(tmpdir, regtest):

    stdout = _setup_first_run(tmpdir)

    _record(stdout, regtest)

    # collect m score stats
    m_score_stat, = [l for l in stdout.split("\n") if "mean m_score" in l]

    # split away log time etc:
    __, __, interesting_m_score_output = m_score_stat.partition("mean m_score")

    # collect s value stats
    s_value_stat, = [l for l in stdout.split("\n") if "mean s_value" in l]

    # split away log time etc:
    __, __, interesting_s_value_output = s_value_stat.partition("mean s_value")

    stdout = subprocess.check_output(
        "pyprophet test_data.txt --apply_scorer=test_data_scorer.bin --target.overwrite --is_test",
        shell=True,
        stderr=subprocess.STDOUT)

    _record(stdout, regtest)

    # collect m score stats
    m_score_stat, = [l for l in stdout.split("\n") if "mean m_score" in l]

    # split away log time etc:
    __, __, interesting_m_score_output2 = m_score_stat.partition("mean m_score")

    # collect s value stats
    s_value_stat, = [l for l in stdout.split("\n") if "mean s_value" in l]

    # split away log time etc:
    __, __, interesting_s_value_output2 = s_value_stat.partition("mean s_value")

    # check if output is consistent over runs:
    assert interesting_m_score_output == interesting_m_score_output2
    assert interesting_s_value_output == interesting_s_value_output2


def test_apply_weights(tmpdir, regtest):

    stdout = _setup_first_run(tmpdir)
    _record(stdout, regtest)

    stdout = subprocess.check_output(
        "pyprophet test_data.txt --apply_weights=test_data_weights.txt --target.overwrite --is_test",
        shell=True,
        stderr=subprocess.STDOUT)

    _record(stdout, regtest)
