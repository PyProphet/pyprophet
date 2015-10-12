# encoding: utf-8
from __future__ import print_function

from pyprophet.stats import get_error_table_from_pvalues_new

import pandas as pd

d = pd.options.display
d.width = 220
d.precision = 6

import numpy as np


def test_0(regtest):
    p_values = np.linspace(0.0, 1., 50)
    p_values = np.append(p_values, np.linspace(1.0 - 0.0001, 1.0, 50))
    df = get_error_table_from_pvalues_new(p_values, use_pfdr=False).df
    print("without correction", file=regtest)
    print(df.head(), file=regtest)
    print(df.tail(), file=regtest)
    df = get_error_table_from_pvalues_new(p_values, use_pfdr=True).df

    # we should see non nan here for FDR:
    print("with correction", file=regtest)
    print(df.head(), file=regtest)
    print(df.tail(), file=regtest)
