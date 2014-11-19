"""
Tests for `brew.combination.combiner` module.
"""

from brew.combination.combiner import Combiner

from brew.combination.rules import max_rule
from brew.combination.rules import min_rule
from brew.combination.rules import mean_rule
from brew.combination.rules import median_rule
from brew.combination.rules import majority_vote_rule


class TestCombiner(object):

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
