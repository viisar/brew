import numpy as np


def majority_vote(delta):
    """ Returns the class label that had the majority of votes
    """

    return delta.sum(axis=1).argmax()
