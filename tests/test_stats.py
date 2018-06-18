# encoding: utf-8
from __future__ import print_function

from pyprophet.stats import to_one_dim_array, pnorm, pemp, pi0est, qvalue, bw_nrd0, lfdr, stat_metrics

import pandas as pd
pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None

import numpy as np
import os
import shutil

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def test_pnorm():
    stat = np.array([0, 1, 3, 2, 0.1, 0.5, 0.6, 0.3, 0.5, 0.6, 0.2, 0.5])
    stat0 = np.array([0.4, 0.2, 0.5, 1, 0.5, 0.7, 0.2, 0.4])

    np.testing.assert_almost_equal(pnorm(stat, stat0),
                                   np.array([9.674763e-01, 2.621760e-02, 0.000000e+00, 5.201675e-09, 9.287418e-01, 4.811347e-01, 3.351438e-01, 7.610205e-01, 4.811347e-01, 3.351438e-01, 8.617105e-01, 4.811347e-01]))


def test_pemp():
    stat = np.array([0, 1, 3, 2, 0.1, 0.5, 0.6, 0.3, 0.5, 0.6, 0.2, 0.5])
    stat0 = np.array([0.4, 0.2, 0.5, 1, 0.5, 0.7, 0.2, 0.4])

    np.testing.assert_almost_equal(pemp(stat, stat0),
                                   np.array([1.0, 0.125, 0.125, 0.125, 1.0, 0.25, 0.25, 0.75, 0.25, 0.25, 0.75, 0.25]))


def test_pi0est(tmpdir):
    os.chdir(tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_lfdr_ref_data.csv")
    shutil.copy(data_path,tmpdir.strpath)

    stat = pd.read_csv('test_lfdr_ref_data.csv', delimiter=',').sort_index(axis=1).sort_values("p")

    # For comparison with R/bioconductor reference implementation
    # np.testing.assert_almost_equal(pi0est(stat['p'], lambda_ = 0.4)['pi0'], 0.6971609)
    # np.testing.assert_almost_equal(pi0est(stat['p'])['pi0'], 0.669926)
    # np.testing.assert_almost_equal(pi0est(stat['p'], lambda_ = np.arange(0.4,1.0,0.05), smooth_log_pi0 = True)['pi0'], 0.6718003)
    # np.testing.assert_almost_equal(pi0est(stat['p'], pi0_method = "bootstrap")['pi0'], 0.6763407)

    np.testing.assert_almost_equal(pi0est(stat['p'], lambda_ = 0.4)['pi0'], 0.697161)
    np.testing.assert_almost_equal(pi0est(stat['p'])['pi0'], 0.6685638)
    np.testing.assert_almost_equal(pi0est(stat['p'], lambda_ = np.arange(0.4,1.0,0.05), smooth_log_pi0 = True)['pi0'], 0.6658949)
    np.testing.assert_almost_equal(pi0est(stat['p'], pi0_method = "bootstrap")['pi0'], 0.6763406)


def test_qvalue(tmpdir, regtest):
    os.chdir(tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_qvalue_ref_data.csv")
    shutil.copy(data_path, tmpdir.strpath)

    stat = pd.read_csv('test_qvalue_ref_data.csv', delimiter=',').sort_index(axis=1).sort_values("p")

    # For comparison with R/bioconductor reference implementation
    np.testing.assert_almost_equal(qvalue(stat['p'], 0.669926026474838, pfdr=False), stat['q_default'].values, decimal=4)
    np.testing.assert_almost_equal(qvalue(stat['p'], 0.669926026474838, pfdr=True), stat['q_pfdr'].values, decimal=4)


def test_bw_nrd0():
    stat = np.array([0, 1, 3, 2, 0.1, 0.5, 0.6, 0.3, 0.5, 0.6, 0.2, 0.5])

    np.testing.assert_almost_equal(bw_nrd0(stat), 0.1736562) # compared with R implementation


def test_lfdr(tmpdir, regtest):
    os.chdir(tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_lfdr_ref_data.csv")
    shutil.copy(data_path, tmpdir.strpath)

    stat = pd.read_csv('test_lfdr_ref_data.csv', delimiter=',').sort_index(axis=1).sort_values("p")

    # For comparison with R/bioconductor reference implementation
    np.testing.assert_almost_equal(lfdr(stat['p'], 0.669926026474838), stat['lfdr_default'].values, decimal=3)
    np.testing.assert_almost_equal(lfdr(stat['p'], 0.669926026474838, monotone = False), stat['lfdr_monotone_false'].values, decimal=3)
    np.testing.assert_almost_equal(lfdr(stat['p'], 0.669926026474838, transf="logit"), stat['lfdr_transf_logit'].values, decimal=3)
    np.testing.assert_almost_equal(lfdr(stat['p'], 0.669926026474838, eps=np.power(10.0,-2)), stat['lfdr_eps'].values, decimal=3)

    print(lfdr(stat['p'], 0.669926026474838), file=regtest)
    print(lfdr(stat['p'], 0.669926026474838, monotone = False), file=regtest)
    print(lfdr(stat['p'], 0.669926026474838, transf="logit"), file=regtest)
    print(lfdr(stat['p'], 0.669926026474838, eps=np.power(10.0,-2)), file=regtest)


def test_stat_metrics(tmpdir, regtest):
    os.chdir(tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_lfdr_ref_data.csv")
    shutil.copy(data_path, tmpdir.strpath)

    stat = pd.read_csv('test_lfdr_ref_data.csv', delimiter=',').sort_values("p")

    print(stat_metrics(to_one_dim_array(stat['p']), 0.669926026474838, False).sort_index(axis=1), file=regtest)
    print(stat_metrics(to_one_dim_array(stat['p']), 0.669926026474838, True).sort_index(axis=1), file=regtest)


def test_random(regtest):
    np.random.seed(1)
    for i in (1, 2, 5, 10, 100):
        for j in (1, 2, 5, 10, 100):
            stat = np.random.random((i,))
            stat0 = np.random.random((j,))
            print(i, j, file=regtest)
            print(pemp(stat, stat0), file=regtest)
