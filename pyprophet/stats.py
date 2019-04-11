from __future__ import division

import numpy as np
import scipy as sp
import pandas as pd
import math
import scipy.stats
import scipy.special
import multiprocessing
import click

from .optimized import (find_nearest_matches as _find_nearest_matches,
                       count_num_positives, single_chromatogram_hypothesis_fast)
from statsmodels.nonparametric.kde import KDEUnivariate
from collections import namedtuple
# from .config import CONFIG

try:
    profile
except NameError:
    profile = lambda x: x


def _ff(a):
    return _find_nearest_matches(*a)


def find_nearest_matches(x, y):
    # num_processes = CONFIG.get("num_processes")
    # if num_processes > 1:
    #     pool = multiprocessing.Pool(processes=num_processes)
    #     batch_size = int(math.ceil(len(y) / num_processes))
    #     parts = [(x, y[i:i + batch_size]) for i in range(0, len(y),
    #                                                      batch_size)]
    #     res = pool.map(_ff, parts)
    #     res_par = np.hstack(res)
    #     return res_par
    return _find_nearest_matches(x, y)


def to_one_dim_array(values, as_type=None):
    """ Converts list or flattens n-dim array to 1-dim array if possible """

    if isinstance(values, (list, tuple)):
        values = np.array(values, dtype=np.float32)
    elif isinstance(values, pd.Series):
        values = values.values
    values = values.flatten()
    assert values.ndim == 1, "values has wrong dimension"
    if as_type is not None:
        return values.astype(as_type)
    return values


@profile
def lookup_values_from_error_table(scores, err_df):
    """ Find matching q-value for each score in 'scores' """
    ix = find_nearest_matches(np.float32(err_df.cutoff.values), np.float32(scores))
    return err_df.pvalue.iloc[ix].values, err_df.svalue.iloc[ix].values, err_df.pep.iloc[ix].values, err_df.qvalue.iloc[ix].values


def posterior_chromatogram_hypotheses_fast(experiment, prior_chrom_null):
    """ Compute posterior probabilities for each chromatogram

    For each chromatogram (each group_id / peptide precursor), all hypothesis of all peaks
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
    pp_values = 1-experiment.df["pep"].values

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


def mean_and_std_dev(values):
    return np.mean(values), np.std(values, ddof=1)


def pnorm(stat, stat0):
    """ [P(X>pi, mu, sigma) for pi in pvalues] for normal distributed stat with
    expectation value mu and std deviation sigma """

    mu, sigma = mean_and_std_dev(stat0)

    stat = to_one_dim_array(stat, np.float64)
    args = (stat - mu) / sigma
    return 1-(0.5 * (1.0 + scipy.special.erf(args / np.sqrt(2.0))))


def pemp(stat, stat0):
    """ Computes empirical values identically to bioconductor/qvalue empPvals """

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


@profile
def pi0est(p_values, lambda_ = np.arange(0.05,1.0,0.05), pi0_method = "smoother", smooth_df = 3, smooth_log_pi0 = False):
    """ Estimate pi0 according to bioconductor/qvalue """

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
        raise click.ClickException("p-values not in valid range [0,1].")
    elif (ll > 1 and ll < 4):
        raise click.ClickException("If lambda_ is not predefined (one value), at least four data points are required.")
    elif (np.min(lambda_) < 0 or np.max(lambda_) >= 1):
        raise click.ClickException("Lambda must be within [0,1)")

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
            raise click.ClickException("pi0_method must be one of 'smoother' or 'bootstrap'.")
    if (pi0<=0):
        raise click.ClickException("The estimated pi0 <= 0. Check that you have valid p-values or use a different range of lambda.")

    return {'pi0': pi0, 'pi0_lambda': pi0_lambda, 'lambda_': lambda_, 'pi0_smooth': pi0Smooth}

@profile
def qvalue(p_values, pi0, pfdr = False):
    p = np.array(p_values)

    qvals_out = p
    rm_na = np.isfinite(p)
    p = p[rm_na]

    if (min(p) < 0 or max(p) > 1):
        raise click.ClickException("p-values not in valid range [0,1].")
    elif (pi0 < 0 or pi0 > 1):
        raise click.ClickException("pi0 not in valid range [0,1].")

    m = len(p)
    u = np.argsort(p)
    v = scipy.stats.rankdata(p,"max")

    if pfdr:
        qvals = (pi0 * m * p) / (v * (1 - np.power((1 - p), m)))
    else:
        qvals = (pi0 * m * p) / v
    
    qvals[u[m-1]] = np.minimum(qvals[u[m-1]], 1)
    for i in list(reversed(range(0,m-2,1))):
        qvals[u[i]] = np.minimum(qvals[u[i]], qvals[u[i + 1]])

    qvals_out[rm_na] = qvals
    return qvals_out


def bw_nrd0(x):
    if len(x) < 2:
        raise click.ClickException("bandwidth estimation requires at least two data points.")

    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    lo = min(hi, iqr/1.34)
    lo = lo or hi or abs(x[0]) or 1

    return 0.9 * lo *len(x)**-0.2


@profile
def lfdr(p_values, pi0, trunc = True, monotone = True, transf = "probit", adj = 1.5, eps = np.power(10.0,-8)):
    """ Estimate local FDR / posterior error probability from p-values according to bioconductor/qvalue """
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
        raise click.ClickException("p-values not in valid range [0,1].")
    elif (pi0 < 0 or pi0 > 1):
        raise click.ClickException("pi0 not in valid range [0,1].")

    # Local FDR method for both probit and logit transformations
    if (transf == "probit"):
        p = np.maximum(p, eps)
        p = np.minimum(p, 1-eps)
        x = scipy.stats.norm.ppf(p, loc=0, scale=1)

        # R-like implementation
        bw = bw_nrd0(x)
        myd = KDEUnivariate(x)
        myd.fit(bw=adj*bw, gridsize = 512)
        splinefit = sp.interpolate.splrep(myd.support, myd.density)
        y = sp.interpolate.splev(x, splinefit)
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

        splinefit = sp.interpolate.splrep(myd.support, myd.density)
        y = sp.interpolate.splev(x, splinefit)
        # myd = density(x, adjust = 1.5) # R reference function
        # mys = smoothspline(x = myd.rx2('x'), y = myd.rx2('y')) # R reference function
        # y = predict(mys, x).rx2('y') # R reference function

        dx = np.exp(x) / np.power((1 + np.exp(x)),2)
        lfdr = (pi0 * dx) / y
    else:
        raise click.ClickException("Invalid local FDR method.")

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
def stat_metrics(p_values, pi0, pfdr):
    num_total = len(p_values)
    num_positives = count_num_positives(p_values)
    num_negatives = num_total - num_positives
    num_null = pi0 * num_total
    tp = num_positives - num_null * p_values
    fp = num_null * p_values
    tn = num_null * (1.0 - p_values)
    fn = num_negatives - num_null * (1.0 - p_values)

    fpr = fp / num_null

    # fdr = fp / num_positives # produces divide by zero warnining
    fdr = np.divide(fp, num_positives, out=np.zeros_like(fp), where=num_positives!=0)

    # fnr = fn / num_negatives # produces divide by zero warnining
    fnr = np.divide(fn, num_negatives, out=np.zeros_like(fn), where=num_negatives!=0)
    
    if pfdr:
        fdr /= (1.0 - (1.0 - p_values) ** num_total)
        fdr[p_values == 0] = 1.0 / num_total

        fnr /= 1.0 - p_values ** num_total
        fnr[p_values == 0] = 1.0 / num_total

    sens = tp / (num_total - num_null)

    sens[sens < 0.0] = 0.0
    sens[sens > 1.0] = 1.0

    fdr[fdr < 0.0] = 0.0
    fdr[fdr > 1.0] = 1.0
    fdr[num_positives == 0] = 0.0

    fnr[fnr < 0.0] = 0.0
    fnr[fnr > 1.0] = 1.0
    fnr[num_positives == 0] = 0.0

    svalues = pd.Series(sens)[::-1].cummax()[::-1]

    return pd.DataFrame({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'fpr': fpr, 'fdr': fdr, 'fnr': fnr, 'svalue': svalues})


@profile
def final_err_table(df, num_cut_offs=51):
    """ Create artificial cutoff sample points from given range of cutoff
    values in df, number of sample points is 'num_cut_offs' """

    cutoffs = df.cutoff.values
    min_ = min(cutoffs)
    max_ = max(cutoffs)
    # extend max_ and min_ by 5 % of full range
    margin = (max_ - min_) * 0.05
    sampled_cutoffs = np.linspace(min_ - margin, max_ + margin, num_cut_offs, dtype=np.float32)

    # find best matching row index for each sampled cut off:
    ix = find_nearest_matches(np.float32(df.cutoff.values), sampled_cutoffs)

    # create sub dataframe:
    sampled_df = df.iloc[ix].copy()
    sampled_df.cutoff = sampled_cutoffs
    # remove 'old' index from input df:
    sampled_df.reset_index(inplace=True, drop=True)

    return sampled_df


@profile
def summary_err_table(df, qvalues=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """ Summary error table for some typical q-values """

    qvalues = to_one_dim_array(qvalues)
    # find best matching fows in df for given qvalues:
    ix = find_nearest_matches(np.float32(df.qvalue.values), qvalues)
    # extract sub table
    df_sub = df.iloc[ix].copy()
    # remove duplicate hits, mark them with None / NAN:
    for i_sub, (i0, i1) in enumerate(zip(ix, ix[1:])):
        if i1 == i0:
            df_sub.iloc[i_sub + 1, :] = None
    # attach q values column
    df_sub.qvalue = qvalues
    # remove old index from original df:
    df_sub.reset_index(inplace=True, drop=True)
    return df_sub[['qvalue','pvalue','svalue','pep','fdr','fnr','fpr','tp','tn','fp','fn','cutoff']]


@profile
def error_statistics(target_scores, decoy_scores, parametric, pfdr, pi0_lambda, pi0_method = "smoother", pi0_smooth_df = 3, pi0_smooth_log_pi0 = False, compute_lfdr = False, lfdr_trunc = True, lfdr_monotone = True, lfdr_transf = "probit", lfdr_adj = 1.5, lfdr_eps = np.power(10.0,-8)):
    """ Takes list of decoy and target scores and creates error statistics for target values """

    target_scores = to_one_dim_array(target_scores)
    target_scores = np.sort(target_scores[~np.isnan(target_scores)])

    decoy_scores = to_one_dim_array(decoy_scores)
    decoy_scores = np.sort(decoy_scores[~np.isnan(decoy_scores)])

    # compute p-values using decoy scores
    if parametric:
        # parametric
        target_pvalues = pnorm(target_scores, decoy_scores)
    else:
        # non-parametric
        target_pvalues = pemp(target_scores, decoy_scores)

    # estimate pi0
    pi0 = pi0est(target_pvalues, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0)

    # compute q-value
    target_qvalues = qvalue(target_pvalues, pi0['pi0'], pfdr)

    # compute other metrics
    metrics = stat_metrics(target_pvalues, pi0['pi0'], pfdr)

    # generate main statistics table
    error_stat = pd.DataFrame({'cutoff': target_scores, 'pvalue': target_pvalues, 'qvalue': target_qvalues, 'svalue': metrics['svalue'], 'tp': metrics['tp'], 'fp': metrics['fp'], 'tn': metrics['tn'], 'fn': metrics['fn'], 'fpr': metrics['fpr'], 'fdr': metrics['fdr'], 'fnr': metrics['fnr']})

    # compute lfdr / PEP
    if compute_lfdr:
        error_stat['pep'] = lfdr(target_pvalues, pi0['pi0'], lfdr_trunc, lfdr_monotone, lfdr_transf, lfdr_adj, lfdr_eps)

    return error_stat, pi0


def find_cutoff(tt_scores, td_scores, cutoff_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0):
    """ Finds cut off target score for specified false discovery rate fdr """

    error_stat, pi0 = error_statistics(tt_scores, td_scores, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, False)
    if not len(error_stat):
        raise click.ClickException("Too little data for calculating error statistcs.")
    i0 = (error_stat.qvalue - cutoff_fdr).abs().idxmin()
    cutoff = error_stat.iloc[i0]["cutoff"]
    return cutoff

