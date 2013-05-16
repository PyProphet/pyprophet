import numpy as np
import scipy.special

class Bunch(dict):
    """A Bunch is a dictionary where keys can be used as attributes"""
    __getattr__ = dict.__getitem__

def bunchify(nested_dict):
    assert isinstance(nested_dict, dict)
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            nested_dict[k] = bunchify(v)
    return Bunch(nested_dict)


def as_one_dim_array(values, as_type=None):
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
    pvalues = as_one_dim_array(pvalues)
    args = (pvalues - mu) / sigma
    return 0.5 * ( 1.0 + scipy.special.erf(args / np.sqrt(2.0)))
