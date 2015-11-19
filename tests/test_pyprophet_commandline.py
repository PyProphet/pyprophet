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
        if "NEEDED" in line:
            continue
        if "time needed" in line:
            continue
        print >> regtest, line


def _expected_output_files():
    return ["test_data_summary_stat.csv",
            "test_data_full_stat.csv",
            "test_data_report.pdf",
            "test_data_cutoffs.txt",
            "test_data_svalues.txt",
            "test_data_qvalues.txt",
            "test_data_dscores_top_target_peaks.txt",
            "test_data_dscores_top_decoy_peaks.txt",
            "test_data_with_dscore.csv",
            "test_data_with_dscore_filtered.csv",
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
    except:
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
    print >> regtest, df
    return 
    print >> regtest
    print >> regtest, f, "contains", len(df), "lines"
    print >> regtest
    if len(lines) > 10:
        print >> regtest, "top 5 lines of", f
        for line in lines[:5]:
            if len(line) > 200:
                line = line[:100] + " ... " + line[-95:]
            print >> regtest, line.rstrip()
        print >> regtest
        print >> regtest, "last 5 lines of", f
        for line in lines[-5:]:
            if len(line) > 200:
                line = line[:100] + " ... " + line[-95:]
            print >> regtest, line.rstrip()
    else:
        for line in lines:
            if len(line) > 200:
                line = line[:100] + " ... " + line[-95:]
            print >> regtest, line.rstrip()


def _dump_digest(full_path):
    """first try: load the pickle as string and compute hexdigests
       --> does not work. cPickle creates slight differences because of memoizing the
           elements of the score_columns attribute of the scorer !
       what works: load pickle and recreate scorer, the _dump (with a "fresh" cPickle, so
       no memoizing) to string than compute hexdigtest
    """
    import hashlib
    import cPickle
    import zlib
    obj = cPickle.loads(zlib.decompress(open(full_path, "rb").read()))
    h = hashlib.sha1()
    h.update(cPickle.dumps(obj))
    return h.hexdigest()


def _run_cmdline(cmdline):
    stdout = cmdline + "\n"
    try:
        stdout += subprocess.check_output(cmdline, shell=True,
                                          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        print >> sys.stderr, e.output
        raise
    return stdout


def _run_pyprophet_to_learn_model(regtest, temp_folder, dump_result_files=False,
        with_probatilities=False, compress=False, sampling_rate=None, use_best=False,
        stat_best=False):
    os.chdir(temp_folder)
    data_path = os.path.join(DATA_FOLDER, "test_data.txt")
    shutil.copy(data_path, temp_folder)
    cmdline = "pyprophet test_data.txt --random_seed=42"
    if with_probatilities:
        cmdline += " --compute.probabilities"
    if compress:
        cmdline += " --target.compress_results"
    if use_best:
        cmdline += " --semi_supervised_learner.use_best"
    if stat_best:
        cmdline += " --semi_supervised_learner.stat_best"
    if sampling_rate is not None:
        cmdline += " --out_of_core --out_of_core.sampling_rate=%f" % sampling_rate
    stdout = _run_cmdline(cmdline)

    for f in _expected_output_files():
        full_path = os.path.join(temp_folder, f)
        assert os.path.exists(full_path)

    if not dump_result_files:
        return stdout

    _dump_output_files(temp_folder, regtest)

    full_path = os.path.join(temp_folder, "test_data_scorer.bin")
    print >> regtest, "hex digtest pickled classifier:", _dump_digest(full_path)
    return stdout


def test_0(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, False)
    _record(stdout, regtest)


def test_1(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, True)
    _record(stdout, regtest)


def test_2(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, True, True)
    _record(stdout, regtest)

    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, True, True, 1.0)
    _record(stdout, regtest)


def test_3(tmpdir, regtest):
    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True, True, True,
                                           use_best=True, stat_best=True)
    _record(stdout, regtest)


def test_apply_classifier(tmpdir, regtest):

    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True)

    _record(stdout, regtest)

    full_path = os.path.join(tmpdir.strpath, "test_data_with_dscore.csv")
    _dump(full_path, regtest)

    full_path = os.path.join(tmpdir.strpath, "test_data_with_dscore_filtered.csv")
    _dump(full_path, regtest)

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
    _remove_output_files(tmpdir)

    """
    the outputs of the previous run and the next run are sligthly different. during learning
    we have multiple iterations to collect the scores for the statistics, when applying the
    weights we only have one run. this should be fixed, I see no reason for this diffference
    and why to use "impured" score values for the statistics.
    """

    stdout = _run_cmdline("pyprophet test_data.txt --apply_weights=test_data_weights.txt "
                          "--target.overwrite --random_seed=42")

    _record(stdout, regtest)
    _dump_output_files(tmpdir, regtest)
    _remove_output_files(tmpdir)

    full_path = os.path.join(tmpdir.strpath, "test_data_scorer.bin")
    d_in_core = _dump_digest(full_path)

    stdout = _run_cmdline("pyprophet test_data.txt --out_of_core --apply_weights=test_data_weights.txt "
                          "--target.overwrite --random_seed=42 --out_of_core.sampling_rate=1.0")

    _record(stdout, regtest)
    _dump_output_files(tmpdir, regtest)

    full_path = os.path.join(tmpdir.strpath, "test_data_scorer.bin")
    d_out_of_core = _dump_digest(full_path)

    assert d_in_core == d_out_of_core


def test_apply_scorer(tmpdir, regtest):

    stdout = _run_pyprophet_to_learn_model(regtest, tmpdir.strpath, True)
    _remove_output_files(tmpdir)

    """TODO: run both with multiple files and / or merge results, same for apply weights !
    """

    stdout = _run_cmdline("pyprophet test_data.txt --apply_scorer=test_data_scorer.bin "
                          "--target.overwrite --random_seed=42")

    _record(stdout, regtest)

    output_files = ["test_data_with_dscore.csv", "test_data_with_dscore_filtered.csv", ]
    _dump_output_files(tmpdir, regtest, names=output_files)
    _remove_output_files(tmpdir, names=output_files)

    # using out of core makes no difference:
    stdout = _run_cmdline("pyprophet test_data.txt --out_of_core "
                          "--apply_scorer=test_data_scorer.bin "
                          "--target.overwrite --random_seed=42")

    _record(stdout, regtest)
    _dump_output_files(tmpdir, regtest, names=output_files)


def test_multiple_input_files(tmpdir, regtest):
    os.chdir(tmpdir.strpath)

    data_path = os.path.join(DATA_FOLDER, "test_data_3.txt")
    shutil.copy(data_path, tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_data_2.txt")
    shutil.copy(data_path, tmpdir.strpath)

    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt --random_seed=42 --target.overwrite")
    _record(stdout, regtest)


def test_multiple_input_files_with_merge(tmpdir, regtest):
    os.chdir(tmpdir.strpath)

    data_path = os.path.join(DATA_FOLDER, "test_data_3.txt")
    shutil.copy(data_path, tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_data_2.txt")
    shutil.copy(data_path, tmpdir.strpath)

    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt --random_seed=42 --target.overwrite "
                          "--multiple_files.merge_results --compute.probabilities")

    _record(stdout, regtest)


def test_out_of_core_multi_input_files(tmpdir, regtest):

    def setup(subfolder):
        f = tmpdir.join(subfolder).strpath
        os.makedirs(f)
        os.chdir(f)
        data_path = os.path.join(DATA_FOLDER, "test_data_3.txt")
        shutil.copy(data_path, f)
        data_path = os.path.join(DATA_FOLDER, "test_data_2.txt")
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
    _dump(full_path, regtest)
    full_path = os.path.join(f1merge, "test_data__summary_stat.csv")
    _dump(full_path, regtest)
    full_path = os.path.join(f2, "test_data__summary_stat.csv")
    _dump(full_path, regtest)
    print >> regtest
    print >> regtest, "this might be different:"
    full_path = os.path.join(f3, "test_data__summary_stat.csv")
    _dump(full_path, regtest)

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


def test_out_of_core_apply_weights(tmpdir, regtest):

    def setup(subfolder="."):
        f = tmpdir.join(subfolder).strpath
        os.makedirs(f)
        os.chdir(f)
        data_path = os.path.join(DATA_FOLDER, "test_data_3.txt")
        shutil.copy(data_path, f)
        data_path = os.path.join(DATA_FOLDER, "test_data_2.txt")
        shutil.copy(data_path, f)
        return f

    setup("out_of_core")
    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt "
                          "--out_of_core "
                          "--out_of_core.sampling_rate=1.0 --random_seed=42 "
                          "--multiple_files.merge_results ")
    _record(stdout, regtest)

    stdout = _run_cmdline("pyprophet test_data_2.txt test_data_3.txt "
                          "--out_of_core "
                          "--apply_weights=test_data__weights.txt "
                          "--out_of_core.sampling_rate=1.0 --random_seed=42 "
                          "--target.overwrite "
                          "--multiple_files.merge_results")

    _record(stdout, regtest)
