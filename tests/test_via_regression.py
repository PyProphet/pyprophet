from __future__ import print_function

import pandas as pd

# set output options for regression tests on a wide terminal
pd.set_option('display.width', 180)

# reduce precision to avoid to sensitive tests because of roundings:
pd.set_option('display.precision', 6)


def test_0(regtest):
    import pyprophet.config
    import pyprophet.pyprophet
    import os.path

    pyprophet.config.CONFIG["is_test"] = True
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data.txt")
    (res, __, tab), __, __ = pyprophet.pyprophet.PyProphet().process_csv(path, "\t")

    print(res, file=regtest)
    print(tab[:10], file=regtest)
    print(tab[-10:], file=regtest)


def test_regression_test_with_probabilities(regtest):
    import pyprophet.config
    import pyprophet.pyprophet
    import os.path

    pyprophet.config.CONFIG["is_test"] = True
    pyprophet.config.CONFIG["compute.probabilities"] = True
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data.txt")
    (res, __, tab), __, __ = pyprophet.pyprophet.PyProphet().process_csv(path, "\t")

    print(res, file=regtest)
    print(tab[:10], file=regtest)
    print(tab[-10:], file=regtest)
