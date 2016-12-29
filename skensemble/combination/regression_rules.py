# TODO: add documentation header

import numpy as np


def max_rule(array):
    """ Outputs the maximum output of all the regressors.

    Parameters
    ----------
    array:  Numpy 1d-array with elements representing each
            regressor output.

    returns The maximum value output by any of the regressors.
    """

    return array.max(axis=1)


def min_rule(array):
    """ Outputs the maximum output of all the regressors.

    Parameters
    ----------
    array:  Numpy 1d-array with elements representing each
            regressor output.

    returns The maximum value output by any of the regressors.
    """

    return array.min(axis=1)


def mean_rule(array):
    """ Outputs the maximum output of all the regressors.

    Parameters
    ----------
    array:  Numpy 1d-array with elements representing each
            regressor output.

    returns The maximum value output by any of the regressors.
    """

    return array.mean(axis=1)


def median_rule(array):
    """ Outputs the maximum output of all the regressors.

    Parameters
    ----------
    array:  Numpy 1d-array with elements representing each
            regressor output.

    returns The maximum value output by any of the regressors.
    """

    return np.median(array, axis=1)


RULE_FUNCTIONS = {
    'max': max_rule,
    'min': min_rule,
    'mean': mean_rule,
    'median': median_rule
}
