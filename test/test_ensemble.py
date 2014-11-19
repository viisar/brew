"""
Tests for `brew.brew` module.
"""

import numpy as np

import brew.Ensemble as Ensemble


class MockClassifier():
    def __init__(self):
        pass

    def fit(X, y):
        pass

    def predict(X):
        pass


class TestEnsemble():

    def test_empty_init(self):
        ens = Ensemble()
        assert ens.classifiers != None
        assert len(ens.classifiers) == 0

    def test_init_one_classifier(self):
        c = MockClassifier()
        ens = Ensemble(classifiers=[c])
        assert len(self.classifiers) == 1

    def test_init_mult_classifiers(self):
        c1 = MockClassifier()
        c2 = MockClassifier()
        c3 = MockClassifier()
        ens = Ensemble(classifiers=[c1,c2,c3])
        assert len(self.classifiers) == 3

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
        assert self.classifiers[0] is c

