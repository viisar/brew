"""
Tests for `brew.combination.combiner` module.
"""

import numpy as np
from brew.combination.combiner import majority_vote
from brew.combination.combiner import max


class TestMajorityVote():

    # rows are classes, columns are classifiers, elements are votes (binary) 
    # with each classifier (column) limited to one vote, that is example.sum(axis=0)
    # should be a all-one vector

    def test_with_one_classifier_one_class(self):
        example = np.array([[1]])
        assert majority_vote(example) == 0
    
    def test_with_one_classifier_mult_classes(self):
        example = np.array([[0],[0],[1]])
        assert majority_vote(example) == 2

    def test_with_mult_classifiers_mult_classes(self):
        example = np.array( [   [0, 0, 0],
                                [1, 0, 1],
                                [0, 1, 0]   ] )
        assert majority_vote(example) == 1


class TestMax():

    # rows are classes, columns are classifiers, elements are posterior probabilities
    # with each classifier (column) having elements sum to 1, that is example.sum(axis=0)
    # should be a all-one vector

    def test_with_one_classifier_one_class(self):
        example = np.array([[1.]])
        assert max(example) == 0
    
    def test_with_one_classifier_mult_classes(self):
        example = np.array([[0.1],[0.5],[0.4]])
        assert max(example) == 1


    def test_with_mult_classifiers_mult_classes(self):
        example = np.array( [   [0.3, 0.6, 0.2],
                                [0.2, 0.2, 0.1],
                                [0.5, 0.2, 0.7]  ] )

        assert max(example) == 2

