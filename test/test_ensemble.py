"""
Tests for `brew.base` module.  """

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from brew.base import transform2votes
from brew.base import Ensemble
from brew.base import EnsembleClassifier

from brew.combination.combiner import Combiner


class MockClassifier():
    def __init__(self):
        pass

    def fit(X, y):
        pass

    def predict(X):
        pass


class TestTransform2Votes():

    def test_one_example_one_class(self):
        sample = np.array([0])
        correct_votes = np.array([[1]])

        assert np.all(transform2votes(sample, 1) == correct_votes)

    def test_one_example_mult_classes(self):
        sample = np.array([2])
        correct_votes = np.array([[0,0,1]])

        assert np.all(transform2votes(sample, 3) == correct_votes)

    def test_mult_examples_one_class(self):
        sample = np.array([0,0,0,0])
        correct_votes = np.array([[1],[1],[1],[1]])

        assert np.all(transform2votes(sample, 1) == correct_votes)

    def test_mult_examples_mult_classes(self):
        sample = np.array([0,2,1,2])

        correct_votes = np.array([  [1,0,0],
                                    [0,0,1],
                                    [0,1,0],
                                    [0,0,1] ])

        assert np.all(transform2votes(sample, 3) == correct_votes)


    def test_complex_example(self):
        sample = np.array([0,1,1,2,3,1,4,3,2])

        correct_votes = np.array([  [1,0,0,0,0],
                                    [0,1,0,0,0],
                                    [0,1,0,0,0],
                                    [0,0,1,0,0],
                                    [0,0,0,1,0],
                                    [0,1,0,0,0],
                                    [0,0,0,0,1],
                                    [0,0,0,1,0],
                                    [0,0,1,0,0]    ])

        assert np.all(transform2votes(sample, 5) == correct_votes)


class TestEnsemble():

    def test_empty_init(self):
        ens = Ensemble()
        assert ens.classifiers != None
        assert len(ens.classifiers) == 0

    def test_init_one_classifier(self):
        c = MockClassifier()
        ens = Ensemble(classifiers=[c])
        assert len(ens.classifiers) == 1

    def test_init_mult_classifiers(self):
        c1 = MockClassifier()
        c2 = MockClassifier()
        c3 = MockClassifier()
        ens = Ensemble(classifiers=[c1,c2,c3])
        assert len(ens.classifiers) == 3

    def test_len_with_empty_init(self):
        ens = Ensemble()
        assert len(ens) == 0

    def test_len_with_one_added(self):
        ens = Ensemble()
        ens.add(MockClassifier())
        assert len(ens) == 1

    def test_len_with_mult_added(self):
        ens = Ensemble()
        ens.add(MockClassifier())
        ens.add(MockClassifier())
        ens.add(MockClassifier())
        assert len(ens) == 3

    def test_add_empty_init(self):
        ens = Ensemble()
        c = MockClassifier()
        ens.add(c)
        assert ens.classifiers[0] is c

    def test_output_with_real_dataset(self):
        pass
        

class TestEnsembleClassifier():

    def test__arguments(self):

        c = MockClassifier()

        pool = Ensemble(classifiers=[c])
        combiner = Combiner(rule='majority_vote')

        model = EnsembleClassifier(ensemble=pool, combiner=combiner)

    def test_none_combiner(self):
        c = MockClassifier()

        pool = Ensemble(classifiers=[c])
        model = EnsembleClassifier(ensemble=pool)

