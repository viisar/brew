import numpy as np

def max(probs):
    """ Returns the class label that was chosen with the maximum confidence by any
    of the classifiers
    """
    return probs.max(axis=1).argmax()

def min(probs):
    return probs.min(axis=1).argmax()

def mean(probs):
    return probs.mean(axis=1).argmax()

def majority_vote(delta):
    """ Returns the class label that had the majority of votes
    """
    return delta.sum(axis=1).argmax()
