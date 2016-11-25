import numpy as np

def entropy_e(oracle):
    """Entropy Measure e"""
    L = oracle.shape[1]
    tmp = np.sum(oracle, axis=1)
    tmp = np.minimum(tmp, L - tmp)

    e = np.mean((1.0 / (L - np.ceil(0.5 * L))) * tmp)

    return e

def kohavi_wolpert_variance(oracle):
    """Kohavi Wolpert Variance"""
    L = oracle.shape[1]
    tmp = np.sum(oracle, axis=1)
    tmp = np.multiply(tmp, L - tmp)

    kw = np.mean((1.0 / (L**2)) * tmp)

    return kw

