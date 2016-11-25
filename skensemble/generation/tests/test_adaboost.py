"""
AdaBoost
"""
import pytest

import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris
from sklearn.base import is_classifier, is_regressor

from skensemble.generation.adaboost import AdaBoost

dtc = DecisionTreeClassifier(max_depth=1)
dtr = DecisionTreeRegressor(max_depth=1)
svc = SVC()
svr = SVR()

iris = load_iris()
X, y = iris['data'], iris['target']
del iris

class TestAdaBoost():

    def test_random_subspace(self):
        gen = AdaBoost(base_estimator=dtc, n_estimators=5)
        gen.fit(X, y)
        ensemble = gen.ensemble
        assert len(ensemble) == 5
        assert set(y) == set(ensemble.classes_)
        assert np.all(ensemble._estimators)

    def test_regression_1(self):
        gen = AdaBoost(base_estimator=dtr, n_estimators=5)
        gen.fit(X, y)
        estimators = gen.ensemble._estimators
        assert np.all(is_regressor(e) for e in estimators)
        assert np.all(isinstance(e, dtr.__class__) for e in estimators)

    def test_regression_2(self):
        gen = AdaBoost(base_estimator=svr, n_estimators=5)
        gen.fit(X, y)
        estimators = gen.ensemble._estimators
        assert np.all(is_regressor(e) for e in estimators)
        assert np.all(isinstance(e, svr.__class__) for e in estimators)

    def test_classification_1(self):
        gen = AdaBoost(base_estimator=dtc, n_estimators=5)
        gen.fit(X, y)
        estimators = gen.ensemble._estimators
        assert np.all(is_classifier(e) for e in estimators)
        assert np.all(isinstance(e, dtc.__class__) for e in estimators)

    def test_classification_2(self):
        gen = AdaBoost(base_estimator=svc, n_estimators=5)
        gen.fit(X, y)
        estimators = gen.ensemble._estimators
        assert np.all(is_classifier(e) for e in estimators)
        assert np.all(isinstance(e, svc.__class__) for e in estimators)

    def test_invalid_base_estimator(self):
        base_estimator = None
        pytest.raises(ValueError, AdaBoost, base_estimator=base_estimator, n_estimators=5)

    def test_no_fit(self):
        gen = AdaBoost(base_estimator=dtc, n_estimators=5)
        with pytest.raises(Exception):
            gen.ensemble

    def test_n_estimators(self):
        for n_estimators in range(1, 20):
            gen = AdaBoost(base_estimator=svc, n_estimators=n_estimators)
            gen.fit(X, y)
            assert len(gen.ensemble) == n_estimators

    def test_zero_n_estimators(self):
        pytest.raises(ValueError, AdaBoost, base_estimator=dtc, n_estimators=0)

    def test_negative_n_estimators(self):
        pytest.raises(ValueError, AdaBoost, base_estimator=dtc, n_estimators=-1)


