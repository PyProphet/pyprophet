import numpy as np
import pandas as pd
import scipy.special

from   util import Bunch

def _as_one_dim_array(values, as_type=None):
    # converst list or flattnes n-dim array to 1-dim array if possible
    if isinstance(values, (list, tuple)):
        values = np.array(values)
    values = values.flatten()
    assert values.ndim == 1, "values has wrong dimension"
    if as_type is not None:
        return values.astype(as_type)
    return values


def pnorm(pvalues, mu, sigma):
    # [P(X>pi, mu, sigma) for pi in pvalues] for normal distr P
    pvalues = _as_one_dim_array(pvalues)
    args = (pvalues - mu) / sigma
    return 0.5 * ( 1.0 + scipy.special.erf(args / np.sqrt(2.0)))


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
    p_values = _as_one_dim_array(p_values)
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

    return Bunch(num_total = conv(num_total),
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


def get_error_table_from_pvalues_new(p_values, lambda_=0.4):

    p_values = np.sort(_as_one_dim_array(p_values))[::-1] # descending

    # estimate FDR, et al:
    num_null = estimate_num_null(p_values, lambda_)
    num = len(p_values)

    p_values = p_values[:,None]
    num_positives = (p_values[:,None] <= p_values[None, :]).sum(axis=0)
    pp = num_positives / num
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

    fpr = false_positives/num_null

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

    return Bunch(df=error_stat,
                 num=num,
                 num_null=num_null,
                 num_alternative=num-num_null)


def get_error_stat_from_null(scores, is_decoy, lambda_=0.4):
    # takes list of scores (eg master score)
    # and list of truth values in is_decoy for indicating negative class

    scores = _as_one_dim_array(scores)
    is_decoy = _as_one_dim_array(is_decoy, np.bool)
    assert len(is_decoy) == len(scores)

    pos_scores = scores[~is_decoy]
    neg_scores = scores[is_decoy]

    pos_scores = np.sort(pos_scores[~np.isnan(pos_scores)])

    num_neg = len(neg_scores)
    mu = np.mean(neg_scores)
    nu = np.std(neg_scores, ddof=1)

    target_pvalues = 1.0 - pnorm(pos_scores, mu, nu)

    result = get_error_table_from_pvalues_new(target_pvalues, lambda_)
    result["target_pvalues"] = target_pvalues
    result["df_error"] = result["df"]
    result["df_error"]["cutoff"] = pos_scores
    del result["df"]

    return result
