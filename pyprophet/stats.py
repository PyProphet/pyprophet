import numpy
from   util import Bunch

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


def fdr_statistics(p_values, cut_off, lambda_ = 0.5):

    # cast p-values to column vector without nan values

    if isinstance(p_values, (list, tuple)):
        p_values = numpy.array(p_values)
    p_values = p_values[~numpy.isnan(p_values)]
    if p_values.ndim == 1:
        p_values = p_values[:, None]

    assert p_values.shape[-1] == 1 and p_values.ndim == 2

    # cast cut off values to row vector
    cut_off_is_float = isinstance(cut_off, float)
    if cut_off_is_float:
        cut_off = numpy.array([cut_off])
    elif isinstance(cut_off, (list, tuple)):
        cut_off = numpy.array(cut_off)

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
        sens = numpy.zeros_like(num_alternative_positive).flatten()
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
