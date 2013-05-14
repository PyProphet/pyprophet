from nose.tools import *

import numpy as np
import pyprophet.stats

from numpy.testing import *


def test_fdr_statistics():

    # for getting the same pseudo random values again:
    np.random.seed(42)
    p_values = np.random.randn(100000)

    # simple cut off value: test i/o type conformance for single cut_off param
    stats = pyprophet.stats.fdr_statistics(p_values, 0.0, lambda_ = 0.4)
    assert_is_instance(stats.sens, float)
    assert_almost_equals(stats.sens, 1.172823, 6)
    assert_is_instance(stats.fdr, float)
    assert_equals(stats.fdr, 0.0)  # must be exactly 0.0 as cutoff is 0.0 !!

    # simple cut off value: test i/o type conformance for mulit cut_off params
    cut_offs = [0.0, 0.5, 1.0]
    stats = pyprophet.stats.fdr_statistics(p_values, cut_offs, lambda_ = 0.4)
    assert_equals(stats.fdr.shape, (3,))
    assert_equals(stats.sens.shape, (3,))

    # regression test on results
    stats = pyprophet.stats.fdr_statistics(p_values, 0.2, lambda_ = 0.4)

    tobe = {'fdr': 0.19832039114178893,
            'num_alternative': 42536.666666666664,
            'num_alternative_negative': -3920.6666666666715,
            'num_alternative_positive': 46457.33333333333,
            'num_negative': 42050.0,
            'num_null': 57463.333333333336,
            'num_null_negative': 45970.66666666667,
            'num_null_positive': 11492.666666666668,
            'num_positive': 57950.0,
            'num_total': 100000.0,
            'sens': 1.0921714599169343}

    for stat_name, value in tobe.items():
        assert_almost_equals(value, stats[stat_name], 6)


def test_estimate_num_null():
    scores = np.arange(0.0, 5.01, 0.2)
    assert pyprophet.stats.estimate_num_null(scores, 0.5) == 46

def test_get_error_table_from_pvalues_new():
    p_values = np.array([ 0.94068207,  0.92209926,  0.89940072,
                          0.87222123, 0.2005352 , 0.13091171,  0.0800891])
    # we use unusual lambda to get stats with variations
    result = pyprophet.stats.get_error_table_from_pvalues_new(p_values, 0.25)

    assert_almost_equals(result.num_null, 5.333333333)
    assert_almost_equals(result.num_alternative, 1.666666666)
    assert_almost_equals(result.num, 7)
    assert_allclose(p_values, result.df.pvalue.values)

    _check(result.df)

def test_get_error_stat_from_null():
    scores = np.arange(0.0, 5.01, 0.2)
    scores[2], scores[3] = scores[3], scores[2]
    is_decoy = [ 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
                 1, 1 ]

    # we use unusual lambda to get stats with variations
    result = pyprophet.stats.get_error_stat_from_null(scores, is_decoy, 0.25)
    p_values_tobe = np.array([ 0.94068207,  0.92209926,  0.89940072,
                               0.87222123, 0.2005352 , 0.13091171,  0.0800891])

    assert_allclose(p_values_tobe, result.target_pvalues)

    _check(result.df_error)

def _check(df):

    assert_sequence_equal(list(df.positive.values), range(7, 0, -1))
    assert_sequence_equal(list(df.negative.values), range(0, 7))

    assert_allclose(df.percentile_positive, [1.000000, 0.857143,
                  0.714286, 0.571429, 0.428571, 0.285714, 0.142857], rtol=1e-5)

    assert_allclose(df.TP, [1.983029, 1.082137, 0.203196, -0.651847,
        1.930479, 1.301804, 0.572858], rtol=1e-5)

    assert_allclose(df.FP,
            [5.016971, 4.917863, 4.796804, 4.651847, 1.069521, 0.698196,
                0.427142], rtol=1e-5)

    assert_allclose(df.TN,
     [0.316362, 0.415471, 0.536529, 0.681487, 4.263812, 4.635138, 4.906191],
     rtol=1e-5)

    assert_allclose(df.FN,
      [-0.316362, 0.584529, 1.463471, 2.318513, -0.263812, 0.364862, 1.093809],
      rtol=1e-5)

    assert_allclose(df.FDR,
                    [0.71671014857142856, 0.8196437866666666,
                        0.95936076800000003, 1.0, 0.35650702222222219,
                        0.3490978933333333, 0.42714186666666665],
      rtol=1e-5)

    assert_allclose(df.qvalue,
                    [0.71671014857142856, 0.71671014857142856,
                        0.71671014857142856, 0.71671014857142856,
                        0.35650702222222219, 0.3490978933333333,
                        0.3490978933333333],
      rtol=1e-5)

    assert_allclose(df.sens,
                    [1.0, 0.64928236800000017, 0.12191769600000002, 0.0, 1.0,
                        0.78108252799999989, 0.34371487999999994],
      rtol=1e-5)

    assert_allclose(df.svalue,
                    [1.0, 1.0, 1.0, 1.0, 1.0, 0.78108252799999989,
                        0.34371487999999994],
      rtol=1e-5)

    assert_allclose(df.FPR,
                    [0.94068206999999993, 0.92209925999999998,
                        0.89940072000000004, 0.87222122999999985, 0.2005352,
                        0.13091170999999999, 0.080089099999999996],
      rtol=1e-5)

