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


example1 = {}
example1['data'] = np.array([   [ 0.04996443,  0.24576966,  0.86503572],
                                [ 0.46393445,  0.23593924,  0.03325572],
                                [ 0.48610112,  0.5182911 ,  0.10170856] ])

example1['max']    = 0
example1['min']    = 2
example1['mean']   = 0
example1['median'] = 2


example2 = {}
example2['data'] = np.array([   [ 0.5700883 ,  0.80026528,  0.18533205],
                                [ 0.32885723,  0.04439391,  0.70580723],
                                [ 0.10105447,  0.15534081,  0.10886072] ])

example2['max']    = 0
example2['min']    = 0
example2['mean']   = 0
example2['median'] = 0


example3 = {}
example3['data'] = np.array([   [ 0.07232759,  0.37269885,  0.67088035],
                                [ 0.05839815,  0.17190474,  0.15302483],
                                [ 0.86927426,  0.45539641,  0.17609482] ])
example3['max']    = 2
example3['min']    = 2
example3['mean']   = 2
example3['median'] = 2


example4 = {}
example4['data'] = np.array([   [ 0.51447044,  0.43834843,  0.2379288 ],
                                [ 0.22804032,  0.20780499,  0.5911353 ],
                                [ 0.25748924,  0.35384658,  0.1709359 ]])

example4['max']    = 1
example4['min']    = 0
example4['mean']   = 0
example4['median'] = 0


example5 = {}
example5['data'] = np.array([   [ 0,  0,  0],
                                [ 0,  1,  0],
                                [ 1,  0,  1]])

example5['majority_vote'] =  2

example6 = {}
example6['data'] = np.array([   [ 1,  0,  0],
                                [ 0,  1,  0],
                                [ 0,  0,  1]])

example6['majority_vote'] =  0


example7 = {}
example7['data'] = np.array([   [ 0,  0,  0],
                                [ 1,  1,  1],
                                [ 0,  0,  0]])

example7['majority_vote'] =  1



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
    def test_example1(self):
        assert max_rule(example1['data']) == example1['max']

    def test_example2(self):
        assert max_rule(example2['data']) == example2['max']

    def test_example3(self):
        assert max_rule(example3['data']) == example3['max']

    def test_example4(self):
        assert max_rule(example4['data']) == example4['max']


class TestMin():
    def test_example1(self):
        assert min_rule(example1['data']) == example1['min']

    def test_example2(self):
        assert min_rule(example2['data']) == example2['min']

    def test_example3(self):
        assert min_rule(example3['data']) == example3['min']

    def test_example4(self):
        assert min_rule(example4['data']) == example4['min']


class TestMean():
    def test_example1(self):
        assert mean_rule(example1['data']) == example1['mean']

    def test_example2(self):
        assert mean_rule(example2['data']) == example2['mean']

    def test_example3(self):
        assert mean_rule(example3['data']) == example3['mean']

    def test_example4(self):
        assert mean_rule(example4['data']) == example4['mean']


class TestMedian():
    def test_example1(self):
        assert median_rule(example1['data']) == example1['median']

    def test_example2(self):
        assert median_rule(example2['data']) == example2['median']

    def test_example3(self):
        assert median_rule(example3['data']) == example3['median']

    def test_example4(self):
        assert median_rule(example4['data']) == example4['median']




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

    def test_example5(self):
        assert majority_vote_rule(example5['data']) == example5['majority_vote']

    def test_example6(self):
        assert majority_vote_rule(example6['data']) == example6['majority_vote']

    def test_example7(self):
        assert majority_vote_rule(example7['data']) == example7['majority_vote']
