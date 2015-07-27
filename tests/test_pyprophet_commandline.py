import os
import subprocess
import shutil
import sys

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


def _run_cmdline(cmdline):
    stdout = cmdline + "\n"
    try:
        stdout += subprocess.check_output(cmdline, shell=True,
                                          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        print >> sys.stderr, e.output
        raise
    return stdout


def _run_pyprophet_to_learn_model(regtest, tmpdir, dump_result_files=False, with_probatilities=False):
    os.chdir(tmpdir.strpath)
    data_path = os.path.join(__here__, "test_data.txt")
    shutil.copy(data_path, tmpdir.strpath)
    if with_probatilities:
        stdout = _run_cmdline("pyprophet test_data.txt --is_test --compute.probabilities")
    else:
        stdout = _run_cmdline("pyprophet test_data.txt --is_test")
    for f in ["test_data_summary_stat.csv", "test_data_full_stat.csv", "test_data_report.pdf",
              "test_data_scorer.bin",
              "test_data_cutoffs.txt",
              "test_data_svalues.txt",
              "test_data_qvalues.txt",
              "test_data_with_dscore.csv",
              "test_data_with_dscore_filtered.csv",
              "test_data_dscores_top_target_peaks.txt",
              "test_data_dscores_top_decoy_peaks.txt",
              ]:
        full_path = os.path.join(tmpdir.strpath, f)
        assert os.path.exists(full_path)

    if not dump_result_files:
        return stdout

    for f in ["test_data_summary_stat.csv", "test_data_full_stat.csv",
              "test_data_cutoffs.txt",
              "test_data_svalues.txt",
              "test_data_qvalues.txt",
              "test_data_dscores_top_target_peaks.txt",
              "test_data_dscores_top_decoy_peaks.txt",
              "test_data_with_dscore.csv",
              "test_data_with_dscore_filtered.csv",
              ]:
        full_path = os.path.join(tmpdir.strpath, f)
        lines = open(full_path, "r").readlines()
        print >> regtest
        print >> regtest, f, "contains", len(lines), "lines"
        print >> regtest
        if len(lines) > 10:
            print >> regtest, "top 5 lines of", f
            for line in lines[:5]:
                if len(line) > 80:
                    line = line[:40] + " ... " + line[-45:]
                print >> regtest, line.rstrip()
            print >> regtest
            print >> regtest, "last 5 lines of", f
            for line in lines[-5:]:
                if len(line) > 80:
                    line = line[:40] + " ... " + line[-45:]
                print >> regtest, line.rstrip()
        else:
            for line in lines:
                if len(line) > 80:
                    line = line[:40] + " ... " + line[-45:]
                print >> regtest, line.rstrip()

    return stdout


def test_0(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir, True, False)
    _record(stdout, regtest)


def test_1(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir, True, True)
    _record(stdout, regtest)


def test_apply_classifier(tmpdir, regtest):

    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir)

    _record(stdout, regtest)

    # collect m score stats
    m_score_stat, = [l for l in stdout.split("\n") if "mean m_score" in l]

    # split away log time etc:
    __, __, interesting_m_score_output = m_score_stat.partition("mean m_score")

    # collect s value stats
    s_value_stat, = [l for l in stdout.split("\n") if "mean s_value" in l]

    # split away log time etc:
    __, __, interesting_s_value_output = s_value_stat.partition("mean s_value")

    stdout = _run_cmdline("pyprophet test_data.txt --apply_scorer=test_data_scorer.bin "
                          "--target.overwrite --is_test")

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

    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir)
    _record(stdout, regtest)

    stdout = _run_cmdline("pyprophet test_data.txt --apply_weights=test_data_weights.txt "
                          "--target.overwrite --is_test")

    _record(stdout, regtest)


def test_multiple_input_files(tmpdir, regtest):
    os.chdir(tmpdir.strpath)

    data_path = os.path.join(__here__, "test_data_3.txt")
    shutil.copy(data_path, tmpdir.strpath)
    data_path = os.path.join(__here__, "test_data_2.txt")
    shutil.copy(data_path, tmpdir.strpath)

    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt --is_test --target.overwrite")
    _record(stdout, regtest)


def test_multiple_input_files_with_merge(tmpdir, regtest):
    os.chdir(tmpdir.strpath)

    data_path = os.path.join(__here__, "test_data_3.txt")
    shutil.copy(data_path, tmpdir.strpath)
    data_path = os.path.join(__here__, "test_data_2.txt")
    shutil.copy(data_path, tmpdir.strpath)

    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt --is_test --target.overwrite "
                          "--multiple_files.merge_results --compute.probabilities")

    _record(stdout, regtest)

