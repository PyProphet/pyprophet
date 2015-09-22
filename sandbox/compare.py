# encoding: utf-8
from __future__ import print_function

import math

import numpy as np
import pandas as pd

"""
We compare the implementaion of FDR in pyprophet to the one from Johan Telemann.

"""

p_values = np.loadtxt("p_values.txt")
p_values = np.sort((p_values))[::-1]

from pyprophet.stats import get_error_table_from_pvalues_new


def calc(pvalues, lamb):
    """ meaning pvalues presorted i descending order"""

    m = len(pvalues)
    pi0 = (pvalues > lamb).sum() / ((1 - lamb)*m)

    pFDR = np.ones(m)
    print("pFDR    y        Pr     fastPow")
    for i in range(m):
        y = pvalues[i]
        Pr = max(1, m - i) / float(m)
        pFDR[i] = (pi0 * y) / (Pr * (1 - math.pow(1-y, m)))
        print(i, pFDR[i], y, Pr, 1.0 - math.pow(1-y, m))


    num_null = pi0*m
    num_alt = m - num_null
    num_negs = np.array(range(m))
    num_pos = m - num_negs
    pp = num_pos / float(m)

    qvalues = np.ones(m)
    qvalues[0] = pFDR[0]
    for i in range(m-1):
        qvalues[i+1] = min(qvalues[i], pFDR[i+1])

    sens = ((1.0 - qvalues) * num_pos) / num_alt
    sens[sens > 1.0] = 1.0

    df = pd.DataFrame(dict(
        pvalue=pvalues,
        qvalue=qvalues,
        FDR=pFDR,
        percentile_positive=pp,
        sens=sens
    ))

    df["svalue"] = df.sens[::-1].cummax()[::-1]

    return df, num_null, m

errstat = get_error_table_from_pvalues_new(p_values, 0.4, True)
fdr_pyprophet = errstat.df["FDR"]
df, __, __ = calc(p_values, 0.4)
fdr_storey = df["FDR"]
fdrs = pd.DataFrame(dict(fdr_pp=fdr_pyprophet, fdr_storey=fdr_storey))
print(fdrs[:34])
print(fdrs[:])
