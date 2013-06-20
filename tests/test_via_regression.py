
def regression_test():
    import pyprophet.config
    import pyprophet.pyprophet
    import os.path
    import numpy 

    config = pyprophet.config.standard_config()
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data.txt")
    res, __, tab = pyprophet.pyprophet.PyProphet().process_csv(path, "\t", config)

    tobe =  [ 7.13743586,-0.29133736,-0.34778976,-1.33578699, numpy.nan,
              numpy.nan, numpy.nan, numpy.nan, numpy.nan]

    numpy.testing.assert_array_almost_equal(res.cutoff.values, tobe)


