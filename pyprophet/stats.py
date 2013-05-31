import numpy as np
import pandas as pd

import util


def secure_divide(denom, nom, replace_zero_division_by=0):
    # avoids division by zero for numpy array
    mask = (nom == 0)
    # prepare division without 'division by zero'
    nom[mask] = 1.0
    result = denom / nom
    # replace 'division by zero' places by default:
    result[mask] = replace_zero_division_by
    # restore input array:
    nom[mask] = 0
    return result


def estimate_num_null(p_values, lambda_):
    # storeys method
    p_values = util.as_one_dim_array(p_values)
    num_null = 1.0 / (1.0-lambda_) * (p_values >= lambda_).sum()
    return num_null


def fdr_statistics(p_values, cut_off, lambda_ = 0.5):

    # cast p-values to column vector without nan values
    if isinstance(p_values, (list, tuple)):
        p_values = np.array(p_values)
    p_values = p_values[~np.isnan(p_values)]
    if p_values.ndim == 1:
        p_values = p_values[:, None]

    assert p_values.shape[-1] == 1 and p_values.ndim == 2

    # cast cut off values to row vector
    cut_off_is_float = isinstance(cut_off, float)
    if cut_off_is_float:
        cut_off = np.array([cut_off])
    elif isinstance(cut_off, (list, tuple)):
        cut_off = np.array(cut_off)

    if cut_off.ndim == 1:
        cut_off = cut_off[None, :]

    assert cut_off.shape[0] == 1 and cut_off.ndim == 2

    # now broadcasting works eg p_values <= cut_off !!!!!!!!!!!!!!!

    # estimated nulls
    num_null = 1.0 / (1.0-lambda_) * (p_values >= lambda_).sum()
    # fdr
    num_positive = (p_values <= cut_off).sum(axis=0)
    num_null_positive = num_null * cut_off
    fdr = secure_divide(num_null_positive, num_positive, 0).flatten()
    # sensitivity
    num_total = len(p_values)
    num_alternative = num_total - num_null
    num_alternative_positive = num_positive - num_null_positive
    if num_alternative == 0:
        sens = np.zeros_like(num_alternative_positive).flatten()
    else:
        sens = num_alternative_positive.flatten() / num_alternative
    # more stat
    num_negative = num_total - num_positive
    num_null_negative= num_null - num_null_positive
    num_alternative_negative = num_negative - num_null_negative

    if cut_off_is_float:
        conv = lambda a: float(a)
    else:
        conv = lambda a: a

    return util.Bunch(num_total = conv(num_total),
                 num_null  = conv(num_null),
                 num_alternative = conv(num_alternative),
                 num_positive = conv(num_positive),
                 num_negative = conv(num_negative),
                 num_null_positive = conv(num_null_positive),
                 num_alternative_positive = conv(num_alternative_positive),
                 num_null_negative = conv(num_null_negative),
                 num_alternative_negative = conv(num_alternative_negative),
                 fdr = conv(fdr),
                 sens = conv(sens))

def get_error_table_using_percentile_positives_new(err_df,
                                                   target_scores,
                                                   num_null):
    num = len(target_scores)
    num_alternative = num - num_null
    target_scores = np.sort(util.as_one_dim_array(target_scores)) # ascending

    #target_scores = target_scores[:, None]
    num_positives = (target_scores[:,None] >= target_scores[None,
        :]).sum(axis=0).flatten()
    num_negatives = num - num_positives
    pp = 1.0 * num_positives / num

    imax = np.abs(err_df.percentile_positive.values[:,None]-pp.T).argmin(axis=0)

    qvalues = err_df.qvalue.iloc[imax].values
    svalues = err_df.svalue.iloc[imax].values
    fdr      = err_df.FDR.iloc[imax].values
    fdr[fdr<0.0] = 0.0
    fdr[fdr>1.0] = 1.0
    fdr[num_positives==0] = 0.0

    fp = fdr * num_positives
    tp = num_positives - fp
    tn = num_null -fp
    fn = num_negatives - tn

    sens =tp / num_alternative
    if num_alternative == 0:
        sens = np.zeros_like(tp)
    sens[sens<0.0] = 0.0
    sens[sens>1.0] = 1.0

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



def lookup_q_values_from_error_table(scores, df):
    scores = util.as_one_dim_array(scores)
    ix = np.abs(scores[None, :] - df.cutoff.values[:,None]).argmin(axis=0)
    return df.qvalue.iloc[ix].values


def final_err_table(df, num_cut_offs = 51):

    cutoffs = df.cutoff.values
    min_ = min(cutoffs)
    max_ = max(cutoffs)
    margin = (max_ - min_) * 0.05

    sampled_cutoffs = np.linspace(min_ - margin, max_ + margin, num_cut_offs)
    ix = np.abs(sampled_cutoffs[None, :] - df.cutoff.values[:,None]).argmin(axis=0)

    sampled_df  = df.iloc[ix]
    sampled_df.cutoff = sampled_cutoffs
    sampled_df.reset_index(inplace=True, drop=True)

    return sampled_df

def summary_err_table(df, qvalues=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
    qvalues = util.as_one_dim_array(qvalues)
    ix = np.abs(qvalues[None, :] - df.qvalue.values[:,None]).argmin(axis=0)
    df_sub = df.iloc[ix]
    for i_sub, (i0, i1) in enumerate(zip(ix, ix[1:])):
        if i1 == i0:
            df_sub.iloc[i_sub+1,:] = None
    df_sub.qvalue = qvalues
    df_sub.reset_index(inplace=True, drop=True)
    return df_sub





def get_error_table_from_pvalues_new(p_values, lambda_=0.4):

    p_values = np.sort(util.as_one_dim_array(p_values))[::-1] # descending

    # estimate FDR, et al:
    num_null = estimate_num_null(p_values, lambda_)
    num = len(p_values)

    p_values = p_values[:,None]
    num_positives = (p_values[:,None] <= p_values[None, :]).sum(axis=0)
    num_negatives = num - num_positives
    pp = 1.0 * num_positives / num
    true_positives = num_positives - num_null * p_values
    false_positives = num_null * p_values
    true_negatives = num_null* (1.0 - p_values)
    false_negatives = num_negatives - num_null* (1.0 - p_values)

    fdr = false_positives/num_positives
    # cut out of range values
    fdr[fdr<0.0] = 0.0
    fdr[fdr>1.0] = 1.0

    sens = true_positives / (num-num_null)
    # cut out of range values
    sens[sens<0.0] = 0.0
    sens[sens>1.0] = 1.0

    if num_null:
        fpr = false_positives/num_null
    else:
        fpr = 0.0 * false_positives

    # assemble data frame
    error_stat = pd.DataFrame(
            dict(pvalue = p_values.flatten(),
                 percentile_positive = pp.flatten(),
                 positive = num_positives.flatten(),
                 negative = num_negatives.flatten(),
                 TP = true_positives.flatten(),
                 FP = false_positives.flatten(),
                 TN = true_negatives.flatten(),
                 FN = false_negatives.flatten(),
                 FDR = fdr.flatten(),
                 sens=sens.flatten(),
                 FPR = fpr.flatten()),
            columns = """pvalue percentile_positive positive negative TP FP
                        TN FN FDR sens FPR""".split()
            )

    # cummin/cummax not available in numpy, so we create them from dataframe
    # here:
    error_stat["qvalue"] = error_stat.FDR.cummin()
    error_stat["svalue"] = error_stat.sens[::-1].cummax()[::-1]

    return util.Bunch(df=error_stat,
                 num=num,
                 num_null=num_null,
                 num_alternative=num-num_null)


def get_error_stat_from_null(pos_scores, neg_scores, lambda_=0.4):
    # takes list of scores (eg master score)
    # and list of truth values in is_decoy for indicating negative class
    pos_scores = util.as_one_dim_array(pos_scores)
    neg_scores = util.as_one_dim_array(neg_scores)

    pos_scores = np.sort(pos_scores[~np.isnan(pos_scores)])

    num_neg = len(neg_scores)
    mu = np.mean(neg_scores)
    nu = np.std(neg_scores, ddof=1)

    print "GET_ERROR_STAT_FROM NULL"
    print len(neg_scores), np.mean(neg_scores), np.std(neg_scores, ddof=1)
    print len(pos_scores), np.mean(pos_scores), np.std(pos_scores, ddof=1)

    #print mu, nu

    target_pvalues = 1.0 - util.pnorm(pos_scores, mu, nu)
    #print target_pvalues[:10]

    result = get_error_table_from_pvalues_new(target_pvalues, lambda_)
    result["target_pvalues"] = target_pvalues
    result["df_error"] = result.df
    df = result.df_error
    df["cutoff"] = pos_scores
    #del result["df"]

    return result

def find_cutoff(pos_scores, neg_scores, lambda_, fdr):

    result = get_error_stat_from_null(pos_scores, neg_scores, lambda_)
    error_table = result.df_error

    error_table.FDR_DIST = (error_table.qvalue - fdr).abs()
    i0 = error_table.FDR_DIST.argmin()  # finds first occurence of minimum
    print error_table.iloc[i0]
    cutoff = error_table.iloc[i0]["cutoff"]

    return cutoff, error_table


