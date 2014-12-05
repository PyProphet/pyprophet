import pdb

def helper_regression(tab):
    import numpy

    ranks = list(tab.peak_group_rank.values)[:10]
    assert ranks == [1, 3, 7, 6, 2, 4, 5, 1, 3, 5], ranks

    ranks = list(tab.peak_group_rank.values)[-10:]
    assert ranks == [12, 2, 15, 9, 8, 11, 4, 16, 14, 10], ranks

    tobe = [5.54973193, -1.59305365, -7.43544856, -4.75434466, -0.99465366]
    numpy.testing.assert_array_almost_equal(tab.d_score.values[:5], tobe)

    tobe = [-3.65461483, -1.52916888, -5.10635037, -4.08469665, -3.46694394]
    numpy.testing.assert_array_almost_equal(tab.d_score.values[-5:], tobe)

    tobe = [8.83676159e-09, 3.73491421e-02,   3.73491421e-02, 3.73491421e-02,
            2.95475612e-02]

    numpy.testing.assert_array_almost_equal(tab.m_score.values[:5], tobe)

    tobe = [ 0.03734914,  0.03734914,  0.03734914,  0.03734914,  0.03734914]
    numpy.testing.assert_array_almost_equal(tab.m_score.values[-5:], tobe)

def test_regression_test():
    import pyprophet.config
    import pyprophet.pyprophet
    import os.path
    import numpy 

    pyprophet.config.CONFIG["is_test"] = True
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data.txt")
    (res, __, tab), __, weights  = pyprophet.pyprophet.PyProphet().process_csv(path, "\t")

    tobe =  [ 7.13743586,-0.29133736,-0.34778976,-1.33578699, None,
              None, None, None, None]

    cutoffs = res.cutoff.values

    assert all(c is None for c in cutoffs[4:])

    numpy.testing.assert_array_almost_equal(cutoffs[:4], tobe[:4])

    assert list(tab.columns)[-3:] == ["d_score", "m_score", "peak_group_rank"]

    helper_regression(tab)

def test_regression_test_with_probabilities():
    import pyprophet.config
    import pyprophet.pyprophet
    import os.path
    import numpy 

    pyprophet.config.CONFIG["is_test"] = True
    pyprophet.config.CONFIG["compute.probabilities"] = True
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data.txt")
    (res, __, tab), __, weights  = pyprophet.pyprophet.PyProphet().process_csv(path, "\t")

    tobe =  [ 7.13743586,-0.29133736,-0.34778976,-1.33578699, None,
              None, None, None, None]

    cutoffs = res.cutoff.values

    assert all(c is None for c in cutoffs[4:])

    numpy.testing.assert_array_almost_equal(cutoffs[:4], tobe[:4])

    assert list(tab.columns)[-6:] == ["d_score", "m_score", "peak_group_rank", "pg_score", "h_score", "h0_score"]

    helper_regression(tab)

    tobe = [0.99977315890393426, 0.00014176369007129515, 8.9297608679281139e-11, 6.3751322006108778e-08, 0.0006064538881158445, 1.092828591717673e-06, 8.4271906976227558e-07, 0.99847440190082992, 0.0016695439684911906, 0.00045795391670980804]
    numpy.testing.assert_array_almost_equal(tab.pg_score.values[:10], tobe)

    tobe = [7.5678685590266921e-07, 0.0015027900527222287, 1.0751036856313322e-07, 3.384859893025539e-06, 4.9223946400897428e-06, 9.3547112941371224e-07, 0.00016557651748619649, 2.6953249420570016e-08, 3.2741200685790881e-07, 1.4786736203618819e-06]
    numpy.testing.assert_array_almost_equal(tab.pg_score.values[-10:], tobe)

    print list(tab.h0_score.values[:10])
    print list(tab.h0_score.values[-10:])

    tobe = [0.99992816301520626, 3.2167376731092143e-08, 2.0259499037913219e-14, 1.4463662768132695e-11, 1.3767348586112663e-07, 2.4793714050816135e-10, 1.9119311640267894e-10, 0.99944120012891957, 2.553783006760038e-06, 6.9965052167373819e-07]
    numpy.testing.assert_array_almost_equal(tab.h_score.values[:10], tobe)

    tobe = [9.2157049910727357e-07, 0.0018327621647134138, 1.3091971532271676e-07, 4.1218940246305116e-06, 5.9942274986881428e-06, 1.1391618961769076e-06, 0.00020166257746127039, 3.2822057083756354e-08, 3.9870290382069074e-07, 1.8006430205363984e-06]
    numpy.testing.assert_array_almost_equal(tab.h_score.values[-10:], tobe)


    tobe = [7.1666690316901841e-05, 7.1666690316901841e-05, 7.1666690316901841e-05, 7.1666690316901841e-05, 7.1666690316901841e-05, 7.1666690316901841e-05, 7.1666690316901841e-05, 0.00055129072141564994, 0.00055129072141564994, 0.00055129072141564994]
    numpy.testing.assert_array_almost_equal(tab.h0_score.values[:10], tobe)

    tobe = [0.8792348247956876, 0.8792348247956876, 0.8792348247956876, 0.8792348247956876, 0.8792348247956876, 0.8792348247956876, 0.8792348247956876, 0.8792348247956876, 0.8792348247956876, 0.8792348247956876]
    numpy.testing.assert_array_almost_equal(tab.h0_score.values[-10:], tobe)




