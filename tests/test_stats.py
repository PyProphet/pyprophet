# encoding: utf-8
from __future__ import print_function

from pyprophet.stats import pi0est, qvalue, pemp, lfdr

import pandas as pd

d = pd.options.display
d.width = 220
d.precision = 6

import numpy as np
import os
import shutil

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def test_0(regtest):
    p_values = np.linspace(0.0, 1., 50)
    p_values = np.append(p_values, np.linspace(1.0 - 0.0001, 1.0, 50))
    pi0 = pi0est(p_values)['pi0']
    df = qvalue(p_values,pi0,use_pfdr=False).df
    print("without correction", file=regtest)
    print(df.head(), file=regtest)
    print(df.tail(), file=regtest)
    df = qvalue(p_values,pi0,use_pfdr=True).df

    # we should see non nan here for FDR:
    print("with correction", file=regtest)
    print(df.head(), file=regtest)
    print(df.tail(), file=regtest)


def test_1():
    stat = np.array([0, 1, 3, 2, 0.1, 0.5, 0.6, 0.3, 0.5, 0.6, 0.2, 0.5])
    stat0 = np.array([0.4, 0.2, 0.5, 1, 0.5, 0.7, 0.2, 0.4])

    np.testing.assert_almost_equal(pemp(stat, stat0),
                                   np.array([1.0, 0.125, 0.125, 0.125, 1.0, 0.25, 0.25, 0.75, 0.25, 0.25, 0.75, 0.25]))

def test_2(tmpdir, regtest):
    os.chdir(tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_lfdr_ref_data.csv")
    shutil.copy(data_path, tmpdir.strpath)

    stat = pd.read_csv('test_lfdr_ref_data.csv', delimiter=',').sort_values("p")

    # For comparison with R/bioconductor reference implementation
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt

    # f = plt.figure(1)
    # plt.subplot(221)
    # plt.plot(lfdr(stat['p'], 0.669926026474838), stat['lfdr_default'].values, '.')

    # plt.subplot(222)
    # plt.plot(lfdr(stat['p'], 0.669926026474838, monotone = False), stat['lfdr_monotone_false'].values, '.')

    # plt.subplot(223)
    # plt.plot(lfdr(stat['p'], 0.669926026474838, transf="logit"), stat['lfdr_transf_logit'].values, '.')

    # plt.subplot(224)
    # plt.plot(lfdr(stat['p'], 0.669926026474838, eps=np.power(10.0,-2)), stat['lfdr_eps'].values, '.')

    # f.savefig("test_lfdr_ref_data.pdf")

    # np.testing.assert_almost_equal(lfdr(stat['p'], 0.669926026474838), stat['lfdr_default'].values)
    # np.testing.assert_almost_equal(lfdr(stat['p'], 0.669926026474838, monotone = False), stat['lfdr_monotone_false'].values)
    # np.testing.assert_almost_equal(lfdr(stat['p'], 0.669926026474838, transf="logit"), stat['lfdr_transf_logit'].values)
    # np.testing.assert_almost_equal(lfdr(stat['p'], 0.669926026474838, eps=np.power(10.0,-2)), stat['lfdr_eps'].values)

    print(lfdr(stat['p'], 0.669926026474838), file=regtest)
    print(lfdr(stat['p'], 0.669926026474838, monotone = False), file=regtest)
    print(lfdr(stat['p'], 0.669926026474838, transf="logit"), file=regtest)
    print(lfdr(stat['p'], 0.669926026474838, eps=np.power(10.0,-2)), file=regtest)

def test_3(tmpdir):
    os.chdir(tmpdir.strpath)
    data_path = os.path.join(DATA_FOLDER, "test_lfdr_ref_data.csv")
    shutil.copy(data_path,tmpdir.strpath)

    stat = pd.read_csv('test_lfdr_ref_data.csv', delimiter=',').sort_values("p")

    # For comparison with R/bioconductor reference implementation
    # np.testing.assert_almost_equal(pi0est(stat['p'], lambda_ = 0.4)['pi0'], 0.6971609)
    # np.testing.assert_almost_equal(pi0est(stat['p'])['pi0'], 0.669926)
    # np.testing.assert_almost_equal(pi0est(stat['p'], lambda_ = np.arange(0.4,1.0,0.05), smooth_log_pi0 = True)['pi0'], 0.6718003)
    # np.testing.assert_almost_equal(pi0est(stat['p'], pi0_method = "bootstrap")['pi0'], 0.6763407)

    np.testing.assert_almost_equal(pi0est(stat['p'], lambda_ = 0.4)['pi0'], 0.6971609)
    np.testing.assert_almost_equal(pi0est(stat['p'])['pi0'], 0.6876971608832817)
    np.testing.assert_almost_equal(pi0est(stat['p'], lambda_ = np.arange(0.4,1.0,0.05), smooth_log_pi0 = True)['pi0'], 0.68769716088327859)
    np.testing.assert_almost_equal(pi0est(stat['p'], pi0_method = "bootstrap")['pi0'], 0.6763407)


def test_random(regtest):
    np.random.seed(1)
    for i in (1, 2, 5, 10, 100):
        for j in (1, 2, 5, 10, 100):
            stat = np.random.random((i,))
            stat0 = np.random.random((j,))
            print(i, j, file=regtest)
            print(pemp(stat, stat0), file=regtest)
