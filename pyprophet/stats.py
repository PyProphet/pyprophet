# encoding: latin-1

from __future__ import division

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

from collections import namedtuple

import numpy as np
import scipy as sp
import pandas as pd

try:
    profile
except NameError:
    profile = lambda x: x

from optimized import (find_nearest_matches as _find_nearest_matches,
                       count_num_positives, single_chromatogram_hypothesis_fast)
import scipy.special
import math
import scipy.stats
from statsmodels.nonparametric.kde import KDEUnivariate

import sys

from config import CONFIG

import multiprocessing

from std_logger import logging

ErrorStatistics = namedtuple("ErrorStatistics", "df num_null num_total")


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
        values = np.array(values, dtype=np.float32)
    elif isinstance(values, pd.Series):
        values = values.values
    values = values.flatten()
    assert values.ndim == 1, "values has wrong dimension"
    if as_type is not None:
        return values.astype(as_type)
    return values


def posterior_pg_prob(dvals, target_scores, decoy_scores, error_stat, number_target_peaks,
                      number_target_pg,
                      given_scores, lambda_):
    """Compute posterior probabilities for each peakgroup

    - Estimate the true distribution by using all target peakgroups above the
      given the cutoff (estimated FDR as given as input). Assume gaussian distribution.

    - Estimate the false/decoy distribution by using all decoy peakgroups.
      Assume gaussian distribution.

    """

    # Note that num_null and num_total are the sum of the
    # cross-validated statistics computed before, therefore the total
    # number of data points selected will be
    #   len(data) /  xeval.fraction * xeval.num_iter
    #
    logging.info("Posterior Probability estimation:")
    logging.info("Estimated number of null %.2f out of a total of %s."
                 % (error_stat.num_null, error_stat.num_total))

    prior_chrom_null = error_stat.num_null / error_stat.num_total
    number_true_chromatograms = (1.0 - prior_chrom_null) * number_target_peaks
    prior_peakgroup_true = number_true_chromatograms / number_target_pg

    logging.info("Prior for a peakgroup: %s" % (number_true_chromatograms / number_target_pg))
    logging.info("Prior for a chromatogram: %s" % (1.0 - prior_chrom_null))
    logging.info("Estimated number of true chromatograms: %s out of %s" %
                 (number_true_chromatograms, number_target_peaks))
    logging.info("Number of target data: %s" % number_target_pg)
    logging.info("")

    # Estimate a suitable cutoff in discriminant score (d_score)
    # target_scores = experiment.get_top_target_peaks().df["d_score"]
    # decoy_scores = experiment.get_top_decoy_peaks().df["d_score"]

    pi0_method = CONFIG.get("final_statistics.pi0_method")
    pi0_smooth_df = CONFIG.get("final_statistics.pi0_smooth_df")
    pi0_smooth_log_pi0 = CONFIG.get("final_statistics.pi0_smooth_log_pi0")
    use_pemp = CONFIG.get("final_statistics.use_pemp")
    use_pfdr = CONFIG.get("final_statistics.use_pfdr")

    estimated_cutoff = find_cutoff(target_scores, decoy_scores, 0.15, lambda_, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, use_pemp, use_pfdr)

    target_scores_above = target_scores[target_scores > estimated_cutoff]

    # Use all decoys and top-peaks of top target chromatograms to
    # parametrically estimate the two distributions

    p_decoy = scipy.stats.norm.pdf(given_scores, np.mean(dvals), np.std(dvals, ddof=1))
    p_target = scipy.stats.norm.pdf(
        given_scores, np.mean(target_scores_above), np.std(target_scores_above, ddof=1))

    # Bayesian inference
    # Posterior probabilities for each peakgroup
    pp_pg_pvalues = p_target * prior_peakgroup_true / (p_target * prior_peakgroup_true
                                                       + p_decoy * (1.0 - prior_peakgroup_true))

    return pp_pg_pvalues


def posterior_chromatogram_hypotheses_fast(experiment, prior_chrom_null):
    """ Compute posterior probabilities for each chromatogram

    For each chromatogram (each transition_group), all hypothesis of all peaks
    being correct (and all others false) as well as the h0 (all peaks are
    false) are computed.

    The prior probability that the  are given in the function

    This assumes that the input data is sorted by tg_num_id

        Args:
            experiment(:class:`data_handling.Multipeptide`): the data of one experiment
            prior_chrom_null(float): the prior probability that any precursor
                is absent (all peaks are false)

        Returns:
            tuple(hypothesis, h0): two vectors that contain for each entry in
            the input dataframe the probabilities for the hypothesis that the
            peak is correct and the probability for the h0
    """

    tg_ids = experiment.df.tg_num_id.values
    pp_values = experiment.df["pg_score"].values

    current_tg_id = tg_ids[0]
    scores = []
    final_result = []
    final_result_h0 = []
    for i in range(tg_ids.shape[0]):

        id_ = tg_ids[i]
        if id_ != current_tg_id:

            # Actual computation for a single transition group (chromatogram)
            prior_pg_true = (1.0 - prior_chrom_null) / len(scores)
            rr = single_chromatogram_hypothesis_fast(
                np.array(scores), prior_chrom_null, prior_pg_true)
            final_result.extend(rr[1:])
            final_result_h0.extend(rr[0] for i in range(len(scores)))

            # Reset for next cycle
            scores = []
            current_tg_id = id_

        scores.append(1.0 - pp_values[i])

    # Last cycle
    prior_pg_true = (1.0 - prior_chrom_null) / len(scores)
    rr = single_chromatogram_hypothesis_fast(np.array(scores), prior_chrom_null, prior_pg_true)
    final_result.extend(rr[1:])
    final_result_h0.extend([rr[0]] * len(scores))

    return final_result, final_result_h0


def pnorm(pvalues, mu, sigma):
    """ [P(X>pi, mu, sigma) for pi in pvalues] for normal distributed P with
    expectation value mu and std deviation sigma """

    pvalues = to_one_dim_array(pvalues, np.float64)
    args = (pvalues - mu) / sigma
    return 0.5 * (1.0 + scipy.special.erf(args / np.sqrt(2.0)))


def pemp(stat, stat0):
    """ Computes empirical values identically to bioconductor/qvalue empPvals
    returns the pvalues of stat based on the distribution of stat0.
    """

    assert len(stat0) > 0
    assert len(stat) > 0

    stat = np.array(stat)
    stat0 = np.array(stat0)

    m = len(stat)
    m0 = len(stat0)

    statc = np.concatenate((stat, stat0))
    v = np.array([True] * m + [False] * m0)
    perm = np.argsort(-statc, kind="mergesort")  # reversed sort, mergesort is stable
    v = v[perm]

    u = np.where(v)[0]
    p = (u - np.arange(m)) / float(m0)

    # ranks can be fractional, we round down to the next integer, ranking returns values starting
    # with 1, not 0:
    ranks = np.floor(scipy.stats.rankdata(-stat)).astype(int) - 1
    p = p[ranks]
    p[p <= 1.0 / m0] = 1.0 / m0

    return p


def mean_and_std_dev(values):
    return np.mean(values), np.std(values, ddof=1)


def get_error_table_using_percentile_positives_new(err_df, target_scores, num_null):
    """ transfer error statistics in err_df for many target scores and given
    number of estimated null hypothesises 'num_null' """

    num_total = len(target_scores)
    num_alternative = num_total - num_null
    target_scores = np.sort(to_one_dim_array(target_scores))  # ascending

    # optimized
    num_positives = count_num_positives(target_scores.astype(np.float64))

    num_negatives = num_total - num_positives

    # the last coertion is needed because depending on the scale of num_total
    # numpy switched to 64 bit floats
    pp = (num_positives.astype(np.float32) / num_total).astype(np.float32)

    # find best matching row in err_df for each percentile_positive in pp:
    imax = find_nearest_matches(err_df.percentile_positive.values, pp)

    qvalues = err_df.qvalue.iloc[imax].values
    svalues = err_df.svalue.iloc[imax].values
    pvalues = err_df.pvalue.iloc[imax].values
    peps = err_df.pep.iloc[imax].values

    fdr = err_df.FDR.iloc[imax].values
    fdr[fdr < 0.0] = 0.0
    fdr[fdr > 1.0] = 1.0
    fdr[num_positives == 0] = 0.0

    fnr = err_df.FNR.iloc[imax].values
    fnr[fnr < 0.0] = 0.0
    fnr[fnr > 1.0] = 1.0
    fnr[num_positives == 0] = 0.0

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
             pvalue=pvalues,
             pep=peps,
             TP=tp,
             FP=fp,
             TN=tn,
             FN=fn,
             FDR=fdr,
             FNR=fnr,
             sens=sens,
             cutoff=target_scores),
        columns="qvalue svalue pvalue pep TP FP TN FN FDR FNR sens cutoff".split(),
    )
    return df_error


@profile
def lookup_values_from_error_table(scores, err_df):
    """ find best matching q-value for each score in 'scores' """
    ix = find_nearest_matches(err_df.cutoff.values, scores)
    return err_df.pvalue.iloc[ix].values, err_df.svalue.iloc[ix].values, err_df.pep.iloc[ix].values, err_df.qvalue.iloc[ix].values


@profile
def final_err_table(df, num_cut_offs=51):
    """ create artificial cutoff sample points from given range of cutoff
    values in df, number of sample points is 'num_cut_offs'"""

    cutoffs = df.cutoff.values
    min_ = min(cutoffs)
    max_ = max(cutoffs)
    # extend max_ and min_ by 5 % of full range
    margin = (max_ - min_) * 0.05
    sampled_cutoffs = np.linspace(min_ - margin, max_ + margin, num_cut_offs, dtype=np.float32)

    # find best matching row index for each sampled cut off:
    ix = find_nearest_matches(df.cutoff.values, sampled_cutoffs)

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
def pi0est(p_values, lambda_ = np.arange(0.05,1.0,0.05), pi0_method = "smoother", smooth_df = 3, smooth_log_pi0 = False):
    """ estimate pi0 with method of storey"""

    # Compare to bioconductor/qvalue reference implementation
    # import rpy2
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    # smoothspline=robjects.r('smooth.spline')
    # predict=robjects.r('predict')

    p = np.array(p_values)

    rm_na = np.isfinite(p)
    p = p[rm_na]
    m = len(p)
    ll = 1
    if isinstance(lambda_, np.ndarray ):
        ll = len(lambda_)
        lambda_ = np.sort(lambda_)

    if (min(p) < 0 or max(p) > 1):
        sys.exit("Error: p-values not in valid range [0,1].")
    elif (ll > 1 and ll < 4):
        sys.exit("Error: If lambda_ is not predefined (one value), at least four data points are required.")
    elif (np.min(lambda_) < 0 or np.max(lambda_) >= 1):
        sys.exit("Error: Lambda must be within [0,1)")

    if (ll == 1):
        pi0 = np.mean(p >= lambda_)/(1 - lambda_)
        pi0_lambda = pi0
        pi0 = np.minimum(pi0, 1)
        pi0Smooth = False
    else:
        pi0 = []
        for l in lambda_:
            pi0.append(np.mean(p >= l)/(1 - l))
        pi0_lambda = pi0

        if (pi0_method == "smoother"):
            if smooth_log_pi0:
                pi0 = np.log(pi0)
                spi0 = sp.interpolate.UnivariateSpline(lambda_, pi0, k=smooth_df)
                pi0Smooth = np.exp(spi0(lambda_))
                # spi0 = smoothspline(lambda_, pi0, df = smooth_df) # R reference function
                # pi0Smooth = np.exp(predict(spi0, x = lambda_).rx2('y')) # R reference function
            else:
                spi0 = sp.interpolate.UnivariateSpline(lambda_, pi0, k=smooth_df)
                pi0Smooth = spi0(lambda_)
                # spi0 = smoothspline(lambda_, pi0, df = smooth_df) # R reference function
                # pi0Smooth = predict(spi0, x = lambda_).rx2('y')  # R reference function
            pi0 = np.minimum(pi0Smooth[ll-1],1)
        elif (pi0_method == "bootstrap"):
            minpi0 = np.percentile(pi0,0.1)
            W = []
            for l in lambda_:
                W.append(np.sum(p >= l))
            mse = (np.array(W) / (np.power(m,2) * np.power((1 - lambda_),2))) * (1 - np.array(W) / m) + np.power((pi0 - minpi0),2)
            pi0 = np.minimum(pi0[np.argmin(mse)],1)
            pi0Smooth = False
        else:
            sys.exit("Error: pi0_method must be one of 'smoother' or 'bootstrap'.")
    if (pi0<=0):
        sys.exit("Error: The estimated pi0 <= 0. Check that you have valid p-values or use a different range of lambda.")

    return {'pi0': pi0, 'pi0_lambda': pi0_lambda, 'lambda_': lambda_, 'pi0_smooth': pi0Smooth}


@profile
def qvalue(p_values, pi0, use_pfdr=False):
    """ estimate error table from p-values with method of storey for estimating fdrs and q-values"""

    # sort descending:
    p_values = np.sort(to_one_dim_array(p_values))[::-1]

    # estimate FDR with storeys method:
    num_total = len(p_values)
    num_null = pi0 * num_total

    # optimized with numpys broadcasting: comparing column vector with row
    # vector yields a matrix with pairwise comparison results.  sum(axis=0)
    # sums up each column:
    num_positives = count_num_positives(p_values)
    num_negatives = num_total - num_positives
    pp = 1.0 * num_positives / num_total
    tp = num_positives - num_null * p_values
    fp = num_null * p_values
    tn = num_null * (1.0 - p_values)
    fn = num_negatives - num_null * (1.0 - p_values)

    fdr = fp / num_positives

    # storey published pFDR as an improvement over FDR,
    # see http://www.genomine.org/papers/directfdr.pdf:

    if use_pfdr:
        fac = 1.0 - (1.0 - p_values) ** num_total
        fdr /= fac
        # if we take the limit p->1 we achieve the following factor:
        fdr[p_values == 0] = 1.0 / num_total

    # cut off values to range 0..1
    fdr[fdr < 0.0] = 0.0
    fdr[fdr > 1.0] = 1.0

    # estimate false non-discovery rate
    fnr = fn / num_negatives

    # storey published pFDR as an improvement over FDR,
    # see http://www.genomine.org/papers/directfdr.pdf:

    if use_pfdr:
        fac = 1.0 - p_values ** num_total
        fnr /= fac
        # if we take the limit p->1 we achieve the following factor:
        fnr[p_values == 0] = 1.0 / num_total

    # cut off values to range 0..1
    fnr[fnr < 0.0] = 0.0
    fnr[fnr > 1.0] = 1.0

    sens = tp / (num_total - num_null)
    # cut off values to range 0..1
    sens[sens < 0.0] = 0.0
    sens[sens > 1.0] = 1.0

    if num_null:
        fpr = fp / num_null
    else:
        fpr = 0.0 * fp

    # assemble statistics as data frame
    df = pd.DataFrame(
        dict(pvalue=p_values.flatten().astype(np.float32),
             percentile_positive=pp.flatten().astype(np.float32),
             positive=num_positives.flatten().astype(np.float32),
             negative=num_negatives.flatten().astype(np.float32),
             TP=tp.flatten().astype(np.float32),
             FP=fp.flatten().astype(np.float32),
             TN=tn.flatten().astype(np.float32),
             FN=fn.flatten().astype(np.float32),
             FDR=fdr.flatten().astype(np.float32),
             FNR=fnr.flatten().astype(np.float32),
             sens=sens.flatten().astype(np.float32),
             FPR=fpr.flatten().astype(np.float32),
             ),
        columns="""pvalue percentile_positive positive negative TP FP
                        TN FN FDR FNR sens FPR""".split(),
    )

    # cummin/cummax not available in numpy, so we create them from dataframe
    # here:
    df["qvalue"] = df.FDR.cummin()
    df["svalue"] = df.sens[::-1].cummax()[::-1]

    return ErrorStatistics(df, num_null, num_total)


def bw_nrd0(x):
    if len(x) < 2:
        sys.exit("Error: bandwidth estimation requires at least two data points.")

    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    lo = min(hi, iqr/1.34)

    if not ((lo == hi) or (lo == abs(x[0])) or (lo == 1)):
        lo = 1

    return 0.9 * lo *len(x)**-0.2


@profile
def lfdr(p_values, pi0, trunc = True, monotone = True, transf = "probit", adj = 1.5, eps = np.power(10.0,-8)):
    """ estimate local FDR / posterior error probability from p-values with method of storey"""
    p = np.array(p_values)

    # Compare to bioconductor/qvalue reference implementation
    # import rpy2
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    # density=robjects.r('density')
    # smoothspline=robjects.r('smooth.spline')
    # predict=robjects.r('predict')

    # Check inputs
    lfdr_out = p
    rm_na = np.isfinite(p)
    p = p[rm_na]

    if (min(p) < 0 or max(p) > 1):
        sys.exit("Error: p-values not in valid range [0,1].")
    elif (pi0 < 0 or pi0 > 1):
        sys.exit("Error: pi0 not in valid range [0,1].")

    # Local FDR method for both probit and logit transformations
    if (transf == "probit"):
        p = np.maximum(p, eps)
        p = np.minimum(p, 1-eps)
        x = scipy.stats.norm.ppf(p, loc=0, scale=1)

        # R-like implementation
        bw = bw_nrd0(x)
        myd = KDEUnivariate(x)
        myd.fit(bw=adj*bw, gridsize = 512)
        y = sp.interpolate.spline(myd.support, myd.density, x)
        # myd = density(x, adjust = 1.5) # R reference function
        # mys = smoothspline(x = myd.rx2('x'), y = myd.rx2('y')) # R reference function
        # y = predict(mys, x).rx2('y') # R reference function

        lfdr = pi0 * scipy.stats.norm.pdf(x) / y
    elif (transf == "logit"):
        x = np.log((p + eps) / (1 - p + eps))

        # R-like implementation
        bw = bw_nrd0(x)
        myd = KDEUnivariate(x)
        myd.fit(bw=adj*bw, gridsize = 512)
        y = sp.interpolate.spline(myd.support, myd.density, x)
        # myd = density(x, adjust = 1.5) # R reference function
        # mys = smoothspline(x = myd.rx2('x'), y = myd.rx2('y')) # R reference function
        # y = predict(mys, x).rx2('y') # R reference function

        dx = np.exp(x) / np.power((1 + np.exp(x)),2)
        lfdr = (pi0 * dx) / y
    else:
        sys.exit("Error: Invalid local FDR method.")

    if (trunc):
        lfdr[lfdr > 1] = 1
    if (monotone):
        lfdr = lfdr[p.ravel().argsort()]
        for i in range(1,len(x)):
            if (lfdr[i] < lfdr[i - 1]):
                lfdr[i] = lfdr[i - 1]
        lfdr = lfdr[scipy.stats.rankdata(p,"min")-1]

    lfdr_out[rm_na] = lfdr
    return lfdr_out


@profile
def get_error_stat_from_null(target_scores, decoy_scores, lambda_, pi0_method = "smoother", pi0_smooth_df = 3, pi0_smooth_log_pi0 = False, use_pemp = False, use_pfdr = False, use_lfdr = False, lfdr_trunc = True, lfdr_monotone = True, lfdr_transf = "probit", lfdr_adj = 1.5, lfdr_eps = np.power(10.0,-8)):
    """ takes list of decoy and target scores and creates error statistics for target values based
    on mean and std dev of decoy scores"""

    decoy_scores = to_one_dim_array(decoy_scores)
    mu, nu = mean_and_std_dev(decoy_scores)

    target_scores = to_one_dim_array(target_scores)
    target_scores = np.sort(target_scores[~np.isnan(target_scores)])

    if use_pemp:
        target_pvalues = pemp(target_scores, decoy_scores)
    else:
        target_pvalues = 1.0 - pnorm(target_scores, mu, nu)

    pi0 = pi0est(target_pvalues, lambda_, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0)
    error_stat = qvalue(target_pvalues, pi0['pi0'], use_pfdr)

    if use_lfdr:
        logging.info("Estimate local false discovery rates (lFDR) / posterior error probabilities (PEP)")
        error_stat.df["pep"] = lfdr(target_pvalues, pi0['pi0'], lfdr_trunc, lfdr_monotone, lfdr_transf, lfdr_adj, lfdr_eps)

    error_stat.df["cutoff"] = target_scores
    return error_stat, target_pvalues, pi0


def find_cutoff(target_scores, decoy_scores, fdr, lambda_, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, use_pemp, use_pfdr):
    """ finds cut off target score for specified false discovery rate fdr """

    error_stat, __, __ = get_error_stat_from_null(target_scores, decoy_scores, lambda_, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, use_pemp, use_pfdr, False)
    if not len(error_stat.df):
        raise Exception("to little data for calculating error statistcs")
    i0 = (error_stat.df.qvalue - fdr).abs().argmin()
    cutoff = error_stat.df.iloc[i0]["cutoff"]
    return cutoff


@profile
def calculate_final_statistics(all_top_target_scores,
                               test_target_scores,
                               test_decoy_scores,
                               lambda_,
                               pi0_method, pi0_smooth_df, 
                               pi0_smooth_log_pi0, use_pemp,
                               use_pfdr=False,
                               lfdr_trunc=True,
                               lfdr_monotone=True,
                               lfdr_transf="probit",
                               lfdr_adj=1.5,
                               lfdr_eps=np.power(10.0,-8)):
    """ estimates error statistics for given samples target_scores and
    decoy scores and extends them to the full table of peak scores in table
    'exp' """

    # estimate error statistics from given samples
    error_stat, target_pvalues, pi0 = get_error_stat_from_null(test_target_scores, test_decoy_scores, lambda_, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, use_pemp, use_pfdr, True, lfdr_trunc, lfdr_monotone, lfdr_transf, lfdr_adj, lfdr_eps)

    # fraction of null hypothesises in sample values
    summed_test_fraction_null = error_stat.num_null / error_stat.num_total

    # transfer statistics from sample set to full set:
    num_top_target = len(all_top_target_scores)

    # most important: transfer estimated number of null hyptothesis:
    num_null_top_target = num_top_target * summed_test_fraction_null

    # now complete error stats based on num_null_top_target:
    raw_error_stat = get_error_table_using_percentile_positives_new(error_stat.df,
                                                                    all_top_target_scores,
                                                                    num_null_top_target)
    return ErrorStatistics(raw_error_stat, error_stat.num_null,
                           error_stat.num_total), target_pvalues, pi0
