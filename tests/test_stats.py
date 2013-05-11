from nose.tools import *

def test_import():
    import pyprophet.stats
    assert pyprophet.stats is not None

def test_fdr_statistics():
    import numpy
    import pyprophet.stats

    # for getting the same pseudo random values again:
    numpy.random.seed(42)
    p_values = numpy.random.randn(100000)

    # simple cut off value: test i/o type conformance for single cut_off param
    stats = pyprophet.stats.fdr_statistics(p_values, 0.0)
    assert_is_instance(stats.sens, float)
    assert_almost_equals(stats.sens, 1.172823, 6)
    assert_is_instance(stats.fdr, float)
    assert_equals(stats.fdr, 0.0)  # must be exactly 0.0 as cutoff is 0.0 !!

    # simple cut off value: test i/o type conformance for mulit cut_off params
    cut_offs = [0.0, 0.5, 1.0]
    stats = pyprophet.stats.fdr_statistics(p_values, cut_offs)
    assert_equals(stats.fdr.shape, (3,))
    assert_equals(stats.sens.shape, (3,))

    # regression test on results
    stats = pyprophet.stats.fdr_statistics(p_values, 0.2)

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


