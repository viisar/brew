# TODO: add documentation header

"""
[1] Kittler, J.; Hatef, M.; Duin, R.P.W.; Matas, J., "On combining
    classifiers," Pattern Analysis and Machine Intelligence, IEEE
    Transactions on , vol.20, no.3, pp.226,239, Mar 1998
"""

import numpy as np


RULE_FUNCTIONS = {
    'max': max_rule,
    'min': min_rule,
    'mean': mean_rule,
    'median': median_rule,
    'majority_vote': majority_vote_rule
}


def _validate_posterior_probs(array):
    """ Checks if array represents a posterior probability ensemble output.

    Sums the rows of all the columns to check if the output probabilities
    of each classifier sum up to one.

    Parameters
    ----------
    array:  Numpy 2d-array with rows representing each class, columns
            representing each classifier.
    """

    if not np.all(a.sum(axis=0) == 1):
        raise ValueError('Input to this combination rule should be a posterior'
                         'probability array with columns summing to one')


def max_rule(array):
    """ Implements the max rule as defined by [1].

    This rule only makes sense if the classifiers output
    the posterior probabilities for each class.

    Parameters
    ----------
    array:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            posterior probabilities.

    returns Numpy 1d-array with each element representing the combination
            of all classifier's output on a single sample using the max rule.
    """

    _validate_posterior_probs(array)
    return array.max(axis=1).argmax()


def min_rule(array):
    """ Implements the min rule as defined by [1].

    This rule only makes sense if the classifiers output
    the posterior probabilities for each class.

    Parameters
    ----------
    array:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            posterior probabilities.

    returns Numpy 1d-array with each element representing the combination
            of all classifier's output on a single sample using the min rule.
    """

    _validate_posterior_probs(array)
    return array.min(axis=1).argmax()


def mean_rule(array):
    """ Implements the first case of the median rule (i.e. mean rule) as defined by [1].

    This rule only makes sense if the classifiers output
    the posterior probabilities for each class.

    Parameters
    ----------
    array:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            posterior probabilities.

    returns Numpy 1d-array with each element representing the combination
            of all classifier's output on a single sample using the mean
            rule.
    """

    _validate_posterior_probs(array)
    return probs.mean(axis=1).argmax()


def median_rule(array):
    """ Implements the second case of the median rule as defined by [1].

    This rule only makes sense if the classifiers output
    the posterior probabilities for each class.

    Parameters
    ----------
    array:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            posterior probabilities.

    returns Numpy 1d-array with each element representing the combination
            of all classifier's output on a single sample using the median
            rule.
    """

    _validate_posterior_probs(array)
    return np.median(array, axis=1).argmax()


def majority_vote_rule(array):
    """ Implements the majority vote rule as defined by [1].

    This rule can always be used, because even if the classifiers output
    posterior probabilities, you can for example, decide to vote for
    the class with the greatest probability. The important thing is to
    transform the classifiers probabilitities/decisions into a matrix
    of votes.

    Parameters
    ----------
    array:  Numpy 2d-array with rows representing each class, columns
            representing each classifier and elements representing
            votes.
    """

    return votes.sum(axis=1).argmax()
