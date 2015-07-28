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


def _run_pyprophet_to_learn_model(regtest, temp_folder, dump_result_files=False, with_probatilities=False):
    os.chdir(temp_folder)
    data_path = os.path.join(__here__, "test_data.txt")
    shutil.copy(data_path, temp_folder)
    if with_probatilities:
        stdout = _run_cmdline("pyprophet test_data.txt --random_seed=42 --compute.probabilities")
    else:
        stdout = _run_cmdline("pyprophet test_data.txt --random_seed=42")
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
        full_path = os.path.join(temp_folder, f)
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
        full_path = os.path.join(temp_folder, f)
        dump(full_path, regtest)

    full_path = os.path.join(temp_folder, "test_data_scorer.bin")
    dump_digest(regtest, full_path, "test_data_scorer.bin")

    return stdout

def dump(full_path, regtest):
    lines = open(full_path, "r").readlines()
    f = os.path.basename(full_path)
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



def test_0(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, False)
    _record(stdout, regtest)


def test_1(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, True)
    _record(stdout, regtest)


def test_apply_classifier(tmpdir, regtest):

    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True)

    _record(stdout, regtest)

    full_path = os.path.join(tmpdir.strpath, "test_data_with_dscore.csv")
    dump(full_path, regtest)

    full_path = os.path.join(tmpdir.strpath, "test_data_with_dscore_filtered.csv")
    dump(full_path, regtest)

    # collect m score stats
    m_score_stat, = [l for l in stdout.split("\n") if "mean m_score" in l]

    # split away log time etc:
    __, __, interesting_m_score_output = m_score_stat.partition("mean m_score")

    # collect s value stats
    s_value_stat, = [l for l in stdout.split("\n") if "mean s_value" in l]

    # split away log time etc:
    __, __, interesting_s_value_output = s_value_stat.partition("mean s_value")

    stdout = _run_cmdline("pyprophet test_data.txt --apply_scorer=test_data_scorer.bin "
                          "--target.overwrite --random_seed=42")

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

    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True)
    _record(stdout, regtest)

    stdout = _run_cmdline("pyprophet test_data.txt --apply_weights=test_data_weights.txt "
                          "--target.overwrite --random_seed=42")

    _record(stdout, regtest)

    for name in ["test_data_summary_stat.csv",
                 "test_data_full_stat.csv",
                 "test_data_report.pdf",
                 "test_data_cutoffs.txt",
                 "test_data_svalues.txt",
                 "test_data_qvalues.txt",
                 "test_data_dscores_top_target_peaks.txt",
                 "test_data_dscores_top_decoy_peaks.txt",
                 "test_data_with_dscore.csv",
                 "test_data_with_dscore_filtered.csv", ]:

        full_path = os.path.join(tmpdir.strpath, name)
        dump(full_path, regtest)

    full_path = os.path.join(tmpdir.strpath, "test_data_scorer.bin")
    dump_digest(regtest, full_path, "test_data_scorer.bin")


def dump_digest(regtest, full_path, name):
    import hashlib
    h = hashlib.sha1()
    h.update(open(full_path, "rb").read())
    print >> regtest
    print >> regtest, "digest", name, ":", h.hexdigest()



def test_multiple_input_files(tmpdir, regtest):
    os.chdir(tmpdir.strpath)

    data_path = os.path.join(__here__, "test_data_3.txt")
    shutil.copy(data_path, tmpdir.strpath)
    data_path = os.path.join(__here__, "test_data_2.txt")
    shutil.copy(data_path, tmpdir.strpath)

    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt --random_seed=42 --target.overwrite")
    _record(stdout, regtest)


def test_multiple_input_files_with_merge(tmpdir, regtest):
    os.chdir(tmpdir.strpath)

    data_path = os.path.join(__here__, "test_data_3.txt")
    shutil.copy(data_path, tmpdir.strpath)
    data_path = os.path.join(__here__, "test_data_2.txt")
    shutil.copy(data_path, tmpdir.strpath)

    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt --random_seed=42 --target.overwrite "
                          "--multiple_files.merge_results --compute.probabilities")

    _record(stdout, regtest)


def test_out_of_core_multi_input_files(tmpdir, regtest):

    def setup(subfolder):
        f = tmpdir.join(subfolder).strpath
        os.makedirs(f)
        os.chdir(f)
        data_path = os.path.join(__here__, "test_data_3.txt")
        shutil.copy(data_path, f)
        data_path = os.path.join(__here__, "test_data_2.txt")
        shutil.copy(data_path, f)
        return f

    f1 = setup("out_of_core_1.0")
    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt "
                          "--out_of_core "
                          "--out_of_core.sampling_rate=1.0 --random_seed=42")
    _record(stdout, regtest)

    f1merge = setup("out_of_core_1.0_merged")
    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt "
                          "--out_of_core "
                          "--out_of_core.sampling_rate=1.0 --random_seed=42 "
                          "--multiple_files.merge_results")

    _record(stdout, regtest)

    f2 = setup("in_core")
    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt --random_seed=42")
    _record(stdout, regtest)

    f2merge = setup("in_core_merged")
    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt --random_seed=42 "
                          "--multiple_files.merge_results")
    _record(stdout, regtest)

    f3 = setup("out_of_core_0.1")
    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt "
                          "--out_of_core --out_of_core.sampling_rate=0.9999999 --random_seed=42")
    _record(stdout, regtest)

    print >> regtest
    print >> regtest, "the next three should be identical:"
    full_path = os.path.join(f1, "test_data__summary_stat.csv")
    dump(full_path, regtest)
    full_path = os.path.join(f1merge, "test_data__summary_stat.csv")
    dump(full_path, regtest)
    full_path = os.path.join(f2, "test_data__summary_stat.csv")
    dump(full_path, regtest)
    print >> regtest
    print >> regtest, "this might be different:"
    full_path = os.path.join(f3, "test_data__summary_stat.csv")
    dump(full_path, regtest)

    compare_folders(f1, f2)
    compare_folders(f1merge, f2merge)


def compare_folders(f1, f2):

    f1files = sorted([os.path.basename(fi) for fi in os.listdir(f1)])
    f2files = sorted([os.path.basename(fi) for fi in os.listdir(f2)])

    assert f1files == f2files

    for name in f1files:
        lines1 = open(os.path.join(f1, name), "r").readlines()
        lines2 = open(os.path.join(f1, name), "r").readlines()
        n1 = len(lines1)
        n2 = len(lines2)
        assert len(lines1) == len(lines2), (name, n1, n2)
        for i, (l1, l2) in enumerate(zip(lines1, lines2)):
            assert l1 == l2, (name, i, l1, l2)
