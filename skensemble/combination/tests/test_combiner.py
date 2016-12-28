"""
Tests for `skensemble.combination.combiner` module.
"""

import pytest

import numpy as np

from skensemble.combination.combiner import Combiner

from skensemble.combination.classification_rules import max_rule as classification_max_rule
from skensemble.combination.classification_rules import min_rule as classification_min_rule
from skensemble.combination.classicication_rules import mean_rule as classification_mean_rule
from skensemble.combination.classification_rules import median_rule as classification_median_rule
from skensemble.combination.classification_rules import majority_vote_rule as classification_majority_vote_rule

from skensemble.combination.regression_rules import max_rule as regression_max_rule
from skensemble.combination.regression_rules import min_rule as regression_min_rule
from skensemble.combination.regression_rules import mean_rule as regression_mean_rule
from skensemble.combination.regression_rules import median_rule as regression_median_rule



data = np.array([[  [ 0.40020766,  0.62850778],
                    [ 0.24844503,  0.35813641],
                    [ 0.35134731,  0.01335582]],

                    [[ 0.7704311 ,  0.3305912 ],
                    [ 0.15983036,  0.51301017],
                    [ 0.06973854,  0.15639863]],

                    [[ 0.04139757,  0.14161513],
                    [ 0.07222793,  0.4083403 ],
                    [ 0.8863745 ,  0.45004458]],

                    [[ 0.04178794,  0.26398028],
                    [ 0.66378837,  0.32354894],
                    [ 0.29442368,  0.41247078]],

                    [[ 0.39572976,  0.32715797],
                    [ 0.32288906,  0.40619746],
                    [ 0.28138118,  0.26664458]]])


class TestCombinerConstructor(object):

    def test_default_rule(self):
        comb = Combiner()
        assert comb.rule == 'majority_vote'
 
    def test_invalid_rule(self):
        with pytest.raises(ValueError) as excinfo:
            comb = Combiner(rule='invalid')

        assert 'invalid argument' in str(excinfo.value)

    def test_rule_that_does_not_allow_weights(self):
        with pytest.raises(ValueError) as excinfo:
            comb = Combiner(rule='max', weights=[1,2,3])
            comb = Combiner(rule='min', weights=[1,2,3])

        assert 'accept weights' in str(excinfo.value)



class TestClassificationCombiner(object):


   
    def test_majority_vote(self):
        comb = Combiner(rule='majority_vote')
        assert comb.rule == majority_vote_rule

    def test_max(self):
        comb = Combiner(rule='max')
        assert comb.rule == max_rule

    def test_min(self):
        comb = Combiner(rule='min')
        assert comb.rule == min_rule

    def test_mean(self):
        comb = Combiner(rule='mean')
        assert comb.rule == mean_rule

    def test_median(self):
        comb = Combiner(rule='median')
        assert comb.rule == median_rule


class TestRegressionCombiner(object):

    def test_invalid_rule(self):
        pass

    def test_default_rule(self):
        comb = Combiner()
        assert comb.rule == majority_vote_rule
    
    def test_majority_vote(self):
        comb = Combiner(rule='majority_vote')
        assert comb.rule == majority_vote_rule

    def test_max(self):
        comb = Combiner(rule='max')
        assert comb.rule == max_rule

    def test_min(self):
        comb = Combiner(rule='min')
        assert comb.rule == min_rule

    def test_mean(self):
        comb = Combiner(rule='mean')
        assert comb.rule == mean_rule

    def test_median(self):
        comb = Combiner(rule='median')
        assert comb.rule == median_rule
