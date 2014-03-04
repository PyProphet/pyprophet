# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")


import numpy as np
import pandas as pd

try:
    profile
except:
    profile = lambda x: x

from optimized import find_nearest_matches as _find_nearest_matches, count_num_positives
import scipy.special
import traceback
import math

from config import CONFIG

import multiprocessing


def _ff(a):
    return _find_nearest_matches(*a)


def find_nearest_matches(x, y):
    num_processes = CONFIG.get("num_processes")
    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
        batch_size = int(math.ceil(len(y) / num_processes))
        parts = [(x, y[i:i + batch_size]) for i in range(0, len(y),
                 batch_size)]
        res = pool.map(_ff, parts)
        res_par = np.hstack(res)
        return res_par
    return _find_nearest_matches(x, y)


def to_one_dim_array(values, as_type=None):
    """ converst list or flattnes n-dim array to 1-dim array if possible"""

    if isinstance(values, (list, tuple)):
        values = np.array(values)
    elif isinstance(values, pd.Series):
        values = values.values
    values = values.flatten()
    assert values.ndim == 1, "values has wrong dimension"
    if as_type is not None:
        return values.astype(as_type)
    return values


def pnorm(pvalues, mu, sigma):
    """ [P(X>pi, mu, sigma) for pi in pvalues] for normal distributed P with
    expectation value mu and std deviation sigma """

    pvalues = to_one_dim_array(pvalues)
    args = (pvalues - mu) / sigma
    return 0.5 * (1.0 + scipy.special.erf(args / np.sqrt(2.0)))


def mean_and_std_dev(values):
    return np.mean(values), np.std(values, ddof=1)


def get_error_table_using_percentile_positives_new(err_df, target_scores, num_null):
    """ transfer error statistics in err_df for many target scores and given
    number of estimated null hypothesises 'num_null' """

    num = len(target_scores)
    num_alternative = num - num_null
    target_scores = np.sort(to_one_dim_array(target_scores))  # ascending

    # optimized 
    num_positives = count_num_positives(target_scores)

    num_negatives = num - num_positives
    pp = num_positives.astype(float) / num

    # find best matching row in err_df for each percentile_positive in pp:
    imax = find_nearest_matches(err_df.percentile_positive, pp)

    qvalues = err_df.qvalue.iloc[imax].values
    svalues = err_df.svalue.iloc[imax].values
    fdr = err_df.FDR.iloc[imax].values
    fdr[fdr < 0.0] = 0.0
    fdr[fdr > 1.0] = 1.0
    fdr[num_positives == 0] = 0.0

    fp = np.round(fdr * num_positives)
    tp = num_positives - fp
    tn = num_null - fp
    fn = num_negatives - tn

    sens = tp / num_alternative
    if num_alternative == 0:
        sens = np.zeros_like(tp)
    sens[sens < 0.0] = 0.0
    sens[sens > 1.0] = 1.0

    df_error = pd.DataFrame(
        dict(qvalue=qvalues,
             svalue=svalues,
             TP=tp,
             FP=fp,
             TN=tn,
             FN=fn,
             FDR=fdr,
             sens=sens,
             cutoff=target_scores),
        columns="qvalue svalue TP FP TN FN FDR sens cutoff".split()
    )

    return df_error


@profile
def lookup_s_and_q_values_from_error_table(scores, err_df):
    """ find best matching q-value foe each score in 'scores' """
    ix = find_nearest_matches(err_df.cutoff, scores)
    return err_df.svalue.iloc[ix].values, err_df.qvalue.iloc[ix].values


@profile
def final_err_table(df, num_cut_offs=51):
    """ create artificial cutoff sample points from given range of cutoff
    values in df, number of sample points is 'num_cut_offs'"""

    cutoffs = df.cutoff.values
    min_ = min(cutoffs)
    max_ = max(cutoffs)
    # extend max_ and min_ by 5 % of full range
    margin = (max_ - min_) * 0.05
    sampled_cutoffs = np.linspace(min_ - margin, max_ + margin, num_cut_offs)

    # find best matching row index for each sampled cut off:
    ix = find_nearest_matches(df.cutoff, sampled_cutoffs)

    # create sub dataframe:
    sampled_df = df.iloc[ix]
    sampled_df.cutoff = sampled_cutoffs
    # remove 'old' index from input df:
    sampled_df.reset_index(inplace=True, drop=True)

    return sampled_df


@profile
def summary_err_table(df, qvalues=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """ summary error table for some typical q-values """

    qvalues = to_one_dim_array(qvalues)
    # find best matching fows in df for given qvalues:
    ix = find_nearest_matches(df.qvalue.values, qvalues)
    # extract sub table
    df_sub = df.iloc[ix]
    # remove duplicate hits, mark them with None / NAN:
    for i_sub, (i0, i1) in enumerate(zip(ix, ix[1:])):
        if i1 == i0:
            df_sub.iloc[i_sub + 1, :] = None
    # attach q values column
    df_sub.qvalue = qvalues
    # remove old index from original df:
    df_sub.reset_index(inplace=True, drop=True)
    return df_sub



@profile
def get_error_table_from_pvalues_new(p_values, lambda_=0.4):
    """ estimate error table from p_values with method of storey for estimating fdrs and q-values
    """

    # sort descending:
    p_values = np.sort(to_one_dim_array(p_values))[::-1]

    # estimate FDR with storeys method:
    num_null = 1.0 / (1.0 - lambda_) * (p_values >= lambda_).sum()
    num = len(p_values)

    # p_values = p_values[:,None]

    # optimized with numpys broadcasting: comparing column vector with row
    # vectory yieds a matrix with pairwaise somparision results.  sum(axis=0)
    # sums up each column:
    num_positives = count_num_positives(p_values)
    num_negatives = num - num_positives
    pp = 1.0 * num_positives / num
    tp = num_positives - num_null * p_values
    fp = num_null * p_values
    tn = num_null * (1.0 - p_values)
    fn = num_negatives - num_null * (1.0 - p_values)

    fdr = fp / num_positives
    # cut off values to range 0..1
    fdr[fdr < 0.0] = 0.0
    fdr[fdr > 1.0] = 1.0

    sens = tp / (num - num_null)
    # cut off values to range 0..1
    sens[sens < 0.0] = 0.0
    sens[sens > 1.0] = 1.0

    if num_null:
        fpr = fp / num_null
    else:
        fpr = 0.0 * fp

    # assemble statisteics as data frame
    error_stat = pd.DataFrame(
        dict(pvalue=p_values.flatten(),
             percentile_positive=pp.flatten(),
             positive=num_positives.flatten(),
             negative=num_negatives.flatten(),
             TP=tp.flatten(),
             FP=fp.flatten(),
             TN=tn.flatten(),
             FN=fn.flatten(),
             FDR=fdr.flatten(),
             sens=sens.flatten(),
             FPR=fpr.flatten()),
        columns="""pvalue percentile_positive positive negative TP FP
                        TN FN FDR sens FPR""".split()
    )

    # cummin/cummax not available in numpy, so we create them from dataframe
    # here:
    error_stat["qvalue"] = error_stat.FDR.cummin()
    error_stat["svalue"] = error_stat.sens[::-1].cummax()[::-1]

    return error_stat, num_null, num


@profile
def get_error_stat_from_null(target_scores, decoy_scores, lambda_):
    """ takes list of decoy and master scores and creates error statistics for target values based
    on mean and std dev of decoy scores"""

    decoy_scores = to_one_dim_array(decoy_scores)
    mu, nu = mean_and_std_dev(decoy_scores)

    target_scores = to_one_dim_array(target_scores)
    target_scores = np.sort(target_scores[~np.isnan(target_scores)])

    target_pvalues = 1.0 - pnorm(target_scores, mu, nu)

    df, num_null, num = get_error_table_from_pvalues_new(target_pvalues, lambda_)
    df["cutoff"] = target_scores
    return df, num_null, num


def find_cutoff(target_scores, decoy_scores, lambda_, fdr):
    """ finds cut off target score for specified false discovery rate fdr """

    df, __, __ = get_error_stat_from_null(target_scores, decoy_scores, lambda_)
    if not len(df):
        raise Exception("to little data for calculating error statistcs")
    i0 = (df.qvalue - fdr).abs().argmin()
    cutoff = df.iloc[i0]["cutoff"]
    return cutoff


@profile
def calculate_final_statistics(all_top_target_scores,
                               test_target_scores,
                               test_decoy_scores,
                               lambda_):
    """ estimates error statistics for given samples target_scores and
    decoy scores and extends them to the full table of peak scores in table
    'exp' """

    # estimate error statistics from given samples
    df, num_null, num_total = get_error_stat_from_null(
        test_target_scores, test_decoy_scores,
        lambda_)

    # fraction of null hypothesises in sample values
    summed_test_fraction_null = float(num_null) / num_total

    # transfer statistics from sample set to full set:
    num_top_target = len(all_top_target_scores)

    # most important: transfer estimted number of null hyptothesis:
    num_null_top_target = num_top_target * summed_test_fraction_null

    # now complete error stats based on num_null_top_target:
    df_raw_stat = get_error_table_using_percentile_positives_new(
        df, all_top_target_scores,
        num_null_top_target)

    return df_raw_stat
