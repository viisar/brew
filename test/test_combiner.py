"""
Tests for `brew.combination.combiner` module.
"""

import numpy as np
from brew.combination.combiner import majority_vote


class TestMajorityVote():

    # rows are classes, columns are classifiers

    def test_with_one_classifier_one_class(self):
        case = np.array([[1]])
        assert majority_vote(case) == 0
    
    def test_with_one_classifier_mult_classes(self):
        case = np.array([[0],[0],[1]])
        assert majority_vote(case) == 2

    def test_with_mult_classifiers_mult_classes(self):
        delta = np.array( [ [0, 0, 0],
                            [1, 0, 1],
                            [0, 1, 0] ] )
        assert majority_vote(delta) == 1

