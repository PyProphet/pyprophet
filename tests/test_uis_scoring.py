# encoding: utf-8
from __future__ import print_function

from pyprophet.uis_scoring import prepare_precursor_bm, prepare_transition_bm, apply_bm, compute_model_fdr

import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_almost_equal

d = pd.options.display
d.width = 220
d.precision = 6

def test_0():
    test_in = {'ms1_pg_score': [0.6], 'ms2_pg_score': [0.8], 'ms2_prec_score': [0.5]}
    tin = pd.DataFrame(data=test_in, index=None)
    test_ref = {'evidence': [0.6, 0.4, 0.5, 0.5], 'hypothesis': [True, False, True, False], 'prior': [0.8, 0.2, 0.8, 0.2]}
    tref = pd.DataFrame(data=test_ref, index=None)

    tout = prepare_precursor_bm(tin)

    assert_frame_equal(tout,tref)

def test_1():
    test_in = {'peptidoforms': ['PEPA|PEPB','PEPB|PEPC','PEPC'], 'prec_pg_score': [0.6,0.6,0.6], 'pg_score': [0.1,0.8,0.6]}
    tin = pd.DataFrame(data=test_in, index=None)
    test_ref = {'evidence': [0.1, 0.1, 0.9, 0.9, 0.2, 0.8, 0.8, 0.2, 0.4, 0.4, 0.6, 0.4], 'hypothesis': ["PEPA", "PEPB", "PEPC", "h0", "PEPA", "PEPB", "PEPC", "h0", "PEPA", "PEPB", "PEPC", "h0"], 'peptidoforms': ["PEPA|PEPB", "PEPA|PEPB", "PEPA|PEPB", "PEPA|PEPB", "PEPB|PEPC", "PEPB|PEPC", "PEPB|PEPC", "PEPB|PEPC", "PEPC", "PEPC", "PEPC", "PEPC"], 'prior': [0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4]}
    tref = pd.DataFrame(data=test_ref, index=None)

    tout = prepare_transition_bm(tin)
    print(tref)
    print(tout)

    assert_frame_equal(tout,tref)

def test_3():
    tin = np.array([0.5, 0.6, 0.8, 0.9, 0.999, 0.1, 0.3])
    tref = np.array([0.2402000, 0.1752500, 0.1003333, 0.0505000, 0.0010000, 0.4001429, 0.3168333])

    tout = compute_model_fdr(tin)

    assert_almost_equal(tout,tref)

def test_4():
    test_in = {'id': 'id0','evidence': [0.1, 0.1, 0.9, 0.9, 0.2, 0.8, 0.8, 0.2, 0.4, 0.4, 0.6, 0.4], 'hypothesis': ["PEPA", "PEPB", "PEPC", "h0", "PEPA", "PEPB", "PEPC", "h0", "PEPA", "PEPB", "PEPC", "h0"], 'peptidoforms': ["PEPA|PEPB", "PEPA|PEPB", "PEPA|PEPB", "PEPA|PEPB", "PEPB|PEPC", "PEPB|PEPC", "PEPB|PEPC", "PEPB|PEPC", "PEPC", "PEPC", "PEPC", "PEPC"], 'prior': [0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4]}
    tin = pd.DataFrame(data=test_in, index=None)
    test_ref = {'id': ['id0','id0','id0','id0'], 'hypothesis': ['PEPA','PEPB','PEPC','h0'],'likelihood_prior': [0.0016, 0.0064, 0.0864, 0.0288],'likelihood_sum': [0.1232, 0.1232, 0.1232, 0.1232],'posterior': [0.012987, 0.051948, 0.701299, 0.233766]}
    tref= pd.DataFrame(data=test_ref, index=None)

    tout = apply_bm(tin)

    #print(tref)
    #print(tout)

    assert_frame_equal(tout[['id','hypothesis','likelihood_prior','likelihood_sum','posterior']],tref[['id','hypothesis','likelihood_prior','likelihood_sum','posterior']])
