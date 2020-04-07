# encoding: utf-8
from __future__ import print_function

from pyprophet.ipf import prepare_precursor_bm, prepare_transition_bm, apply_bm, compute_model_fdr

import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal

pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None

def test_0():
    test_in = pd.DataFrame({'feature_id': [0], 'ms1_precursor_pep': [0.4], 'ms2_peakgroup_pep': [0.2], 'ms2_precursor_pep': [0.5]})
    test_ref = pd.DataFrame(data={'evidence': [0.6, 0.4, 0.5, 0.5], 'feature_id': [0, 0, 0, 0], 'hypothesis': [True, False, True, False], 'prior': [0.8, 0.2, 0.8, 0.2]}, index=None)

    test_out = prepare_precursor_bm(test_in)

    assert_frame_equal(test_out[['feature_id', 'prior', 'evidence', 'hypothesis']].reset_index(drop=True),test_ref[['feature_id', 'prior', 'evidence', 'hypothesis']].reset_index(drop=True))

def test_1():
    tin = np.array([0.5, 0.4, 0.2, 0.1, 0.001, 0.9, 0.7])
    tref = np.array([0.2402000, 0.1752500, 0.1003333, 0.0505000, 0.0010000, 0.4001429, 0.3168333])

    tout = compute_model_fdr(tin)

    assert_almost_equal(tout,tref)

def test_2():
    tin = np.array([0.5, 0.4, 0.2, 0.1, 0.001, 0.9, 0.7, 0.001, 0, 0.3, 0.12, 0.4, 0.1111, 0.2222, 0.88887, 1.0, 0.0000000000001])
    tref = np.array([ 1.81176923e-01, 1.54608333e-01, 6.66375000e-02, 2.04000000e-02, 5.00000000e-04, 3.02760625e-01, 2.18235714e-01, 5.00000000e-04, 0.00000000e+00, 1.05530000e-01, 4.75857143e-02, 1.54608333e-01, 3.55166667e-02, 8.39222222e-02, 2.62944667e-01, 3.43774706e-01, 5.00155473e-14])

    tout = compute_model_fdr(tin)

    assert_almost_equal(tout,tref)

def test_3():
    test_in = {'feature_id': 'id0','evidence': [0.1, 0.1, 0.9, 0.9, 0.2, 0.8, 0.8, 0.2, 0.4, 0.4, 0.6, 0.4], 'hypothesis': [1, 2, 3, -1, 1, 2, 3, -1, 1, 2, 3, -1], 'prior': [0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4]}
    tin = pd.DataFrame(data=test_in, index=None)
    test_ref = {'feature_id': ['id0','id0','id0','id0'], 'hypothesis': [-1, 1, 2, 3,],'likelihood_prior': [0.0288, 0.0016, 0.0064, 0.0864],'likelihood_sum': [0.1232, 0.1232, 0.1232, 0.1232],'posterior': [0.233766, 0.012987, 0.051948, 0.701299]}
    tref= pd.DataFrame(data=test_ref, index=None)

    tout = apply_bm(tin)

    print(tref)
    print(tout)

    assert_frame_equal(tout[['feature_id','hypothesis','likelihood_prior','likelihood_sum','posterior']],tref[['feature_id','hypothesis','likelihood_prior','likelihood_sum','posterior']])
