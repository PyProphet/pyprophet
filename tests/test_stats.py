# encoding: utf-8
from __future__ import print_function

from pyprophet.stats import get_error_table_from_pvalues_new, pemp

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

def test_1():
    stat = np.array([0,1,3,2,0.1,0.5,0.6,0.3,0.5,0.6,0.2,0.5])
    stat0 = np.array([0.4,0.2,0.5,1,0.5,0.7,0.2,0.4])

    np.testing.assert_almost_equal(pemp(stat,stat0), np.array([1.0, 0.125, 0.125, 0.125, 1.0, 0.25, 0.25, 0.75, 0.25, 0.25, 0.75, 0.25]))

