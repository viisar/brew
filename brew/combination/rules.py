import numpy as np


def max_rule(probs):
    """ Returns the class label that was chosen with the maximum confidence by any
    of the classifiers
    """
    return probs.max(axis=1).argmax()

def min_rule(probs):
    return probs.min(axis=1).argmax()

def mean_rule(probs):
    return probs.mean(axis=1).argmax()

def median_rule(probs):
    # numpy array has no median method
    return np.median(probs, axis=1).argmax()

def majority_vote_rule(votes):
    """ Returns the class label that had the majority of votes
    """
    return votes.sum(axis=1).argmax()
