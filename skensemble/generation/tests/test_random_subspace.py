"""
RandomSubspace
"""
import pytest

import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris
from sklearn.base import is_classifier, is_regressor

from skensemble.generation.random_subspace import RandomSubspace

dtc = DecisionTreeClassifier()
dtr = DecisionTreeRegressor()
svc = SVC()
svr = SVR()

iris = load_iris()
X, y = iris['data'], iris['target']
del iris

class TestRandomSubspace():

    def test_random_subspace(self):
        gen = RandomSubspace(base_estimator=dtc, n_estimators=5)
        gen.fit(X, y)
        ensemble = gen.ensemble
        assert len(ensemble) == 5
        assert set(y) == set(ensemble.classes_)
        assert np.all(ensemble._estimators)

    def test_random_subspace_regression_1(self):
        gen = RandomSubspace(base_estimator=dtr, n_estimators=5)
        gen.fit(X, y)
        estimators = gen.ensemble._estimators
        assert np.all(is_regressor(e) for e in estimators)
        assert np.all(isinstance(e, dtr.__class__) for e in estimators)

    def test_random_subspace_regression_2(self):
        gen = RandomSubspace(base_estimator=svr, n_estimators=5)
        gen.fit(X, y)
        estimators = gen.ensemble._estimators
        assert np.all(is_regressor(e) for e in estimators)
        assert np.all(isinstance(e, svr.__class__) for e in estimators)

    def test_random_subspace_classification_1(self):
        gen = RandomSubspace(base_estimator=dtc, n_estimators=5)
        gen.fit(X, y)
        estimators = gen.ensemble._estimators
        assert np.all(is_classifier(e) for e in estimators)
        assert np.all(isinstance(e, dtc.__class__) for e in estimators)

    def test_random_subspace_classification_2(self):
        gen = RandomSubspace(base_estimator=svc, n_estimators=5)
        gen.fit(X, y)
        estimators = gen.ensemble._estimators
        assert np.all(is_classifier(e) for e in estimators)
        assert np.all(isinstance(e, svc.__class__) for e in estimators)

    def test_random_subspace_invalid_base_estimator(self):
        base_estimator = None
        pytest.raises(ValueError, RandomSubspace, base_estimator=base_estimator, n_estimators=5)

    def test_random_subspace_no_fit(self):
        gen = RandomSubspace(base_estimator=dtc, n_estimators=5)
        with pytest.raises(Exception):
            gen.ensemble

    def test_random_subspace_n_estimators(self):
        for n_estimators in range(1, 20):
            gen = RandomSubspace(base_estimator=svc, n_estimators=n_estimators)
            gen.fit(X, y)
            assert len(gen.ensemble) == n_estimators

    def test_random_subspace_zero_n_estimators(self):
        pytest.raises(ValueError, RandomSubspace, base_estimator=dtc, n_estimators=0)

    def test_random_subspace_negative_n_estimators(self):
        pytest.raises(ValueError, RandomSubspace, base_estimator=dtc, n_estimators=-1)

    def test_random_subspace_max_features(self):
        gen = RandomSubspace(base_estimator=dtc, n_estimators=5, max_features=2)
        gen.fit(X, y)
        assert np.all([len(feats) <= 2 for feats in gen.estimators_features])

    def test_random_subspace_max_features_float(self):
        gen = RandomSubspace(base_estimator=dtc, n_estimators=5, max_features=0.5)
        gen.fit(X, y)
        assert np.all([len(feats) <= X.shape[1]/2. for feats in gen.estimators_features])

    def test_random_subspace_zero_max_features(self):
        pytest.raises(ValueError, RandomSubspace, base_estimator=dtc, n_estimators=5, max_features=0)

    def test_random_subspace_negative_max_features(self):
        pytest.raises(ValueError, RandomSubspace, base_estimator=dtc, n_estimators=5, max_features=-1)


