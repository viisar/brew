"""
Tests for `brew.combination.rules` module.
"""

import numpy as np

# rules that expect the posterior probabilities
from brew.combination.rules import max_rule
from brew.combination.rules import min_rule
from brew.combination.rules import mean_rule
from brew.combination.rules import median_rule

# rules that expect the decision votes
from brew.combination.rules import majority_vote_rule


class TestAllProbRules():

    # rows are classes, columns are classifiers, elements are posterior probabilities
    # with each classifier (column) having elements sum to 1, that is example.sum(axis=0)
    # should be a all-one vector

    def test_with_one_classifier_one_class(self):
        example = np.array([[1.]])
        assert max_rule(example) == 0
        assert min_rule(example) == 0
        assert mean_rule(example) == 0
        assert median_rule(example) == 0
    
    def test_with_mult_classifiers_one_class(self):
        example = np.array([[0.3, 0.5, 0.1]])
        assert max_rule(example) == 0
        assert min_rule(example) == 0
        assert mean_rule(example) == 0
        assert median_rule(example) == 0
    
    def test_with_one_classifier_mult_classes(self):
        example = np.array([[0.1],[0.5],[0.4]])
        assert max_rule(example) == 1
        assert min_rule(example) == 1
        assert mean_rule(example) == 1
        assert median_rule(example) == 1


class TestMax():

    def test_with_mult_classifiers_mult_classes(self):
        example = np.array( [   [0.3, 0.6, 0.2],
                                [0.2, 0.2, 0.1],
                                [0.5, 0.2, 0.7]  ] )

        assert max_rule(example) == 2


class TestMin():
    def test_with_mult_classifiers_mult_classes(self):
        example = np.array( [   [0.5, 0.15, 0.1],
                                [0.3, 0.25, 0.7],
                                [0.2,  0.6, 0.2]  ] )

        assert min_rule(example) == 1


class TestMean():
    def test_with_mult_classifiers_mult_classes(self):
        example = np.array( [   [0.3, 0.6, 0.2],
                                [0.2, 0.2, 0.1],
                                [0.5, 0.2, 0.7]  ] )

        assert mean_rule(example) == 2

class TestMedian():
    def test_with_mult_classifiers_mult_classes(self):
        example = np.array( [   [0.3, 0.6, 0.2],
                                [0.5, 0.2, 0.1],
                                [0.2, 0.2, 0.7]  ] )

        assert median_rule(example) == 0


class TestMajorityVote():

    # rows are classes, columns are classifiers, elements are votes (binary) 
    # with each classifier (column) limited to one vote, that is example.sum(axis=0)
    # should be a all-one vector

    def test_with_one_classifier_one_class(self):
        example = np.array([[1]])
        assert majority_vote_rule(example) == 0
    
    def test_with_one_classifier_mult_classes(self):
        example = np.array([[0],[0],[1]])
        assert majority_vote_rule(example) == 2

    def test_with_mult_classifiers_mult_classes(self):
        example = np.array( [   [0, 0, 0],
                                [1, 0, 1],
                                [0, 1, 0]   ] )
        assert majority_vote_rule(example) == 1
