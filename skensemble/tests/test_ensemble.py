"""
See: Kuncheva L.I., C.J. Whitaker. Measures of diversity in classifier ensembles, Machine Learning , 51 , 2003, 181-207,
"""
import pytest

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris

from skensemble.ensemble import Ensemble
from skensemble.ensemble import ENSEMBLE_TYPE_CLASSIFIERS
from skensemble.ensemble import ENSEMBLE_TYPE_REGRESSORS
from skensemble.ensemble import output2labels
from skensemble.ensemble import is_ensemble_of_classifiers, is_ensemble_of_regressors

data = load_iris()
X = data['data']
y = data['target']

class TestEnsemble():

    def test_ensemble(self):
        ensemble = Ensemble()
        assert True

    def test_ensemble_infer_type_classifiers_1(self):
        ensemble = Ensemble([DecisionTreeClassifier()])
        assert ensemble.type_ == ENSEMBLE_TYPE_CLASSIFIERS
        assert is_ensemble_of_classifiers(ensemble)

    def test_ensemble_infer_type_classifiers_2(self):
        ensemble = Ensemble()
        ensemble.append(DecisionTreeClassifier())
        assert ensemble.type_ == ENSEMBLE_TYPE_CLASSIFIERS
        assert is_ensemble_of_classifiers(ensemble)

    def test_ensemble_infer_type_regressors_1(self):
        ensemble = Ensemble([DecisionTreeRegressor()])
        assert ensemble.type_ == ENSEMBLE_TYPE_REGRESSORS
        assert is_ensemble_of_regressors(ensemble)

    def test_ensemble_infer_type_regressors_2(self):
        ensemble = Ensemble()
        ensemble.append(DecisionTreeRegressor())
        assert ensemble.type_ == ENSEMBLE_TYPE_REGRESSORS
        assert is_ensemble_of_regressors(ensemble)

    def test_ensemble_mixing_estimators_1(self):
        ensemble = Ensemble()
        ensemble.append(DecisionTreeRegressor())
        pytest.raises(ValueError, ensemble.append, DecisionTreeClassifier())

    def test_ensemble_mixing_estimators_2(self):
        ensemble = Ensemble()
        ensemble.append(DecisionTreeClassifier())
        pytest.raises(ValueError, ensemble.append, DecisionTreeRegressor())

    def test_ensemble_mixing_estimators_3(self):
        ensemble = Ensemble([DecisionTreeClassifier()])
        pytest.raises(ValueError, ensemble.append, DecisionTreeRegressor())

    def test_ensemble_extend(self):
        ensemble = Ensemble([DecisionTreeClassifier()])
        ensemble.extend([DecisionTreeClassifier(), DecisionTreeClassifier()])
        assert len(ensemble) == 3

    def test_ensemble_extend_error(self):
        ensemble = Ensemble([DecisionTreeClassifier()])
        pytest.raises(ValueError, ensemble.extend, 
                [DecisionTreeClassifier(), DecisionTreeClassifier(), DecisionTreeRegressor()])

    def test_ensemble_type_classifiers(self):
        ensemble = Ensemble(type_=ENSEMBLE_TYPE_CLASSIFIERS)
        assert ensemble.type_ == ENSEMBLE_TYPE_CLASSIFIERS

    def test_ensemble_type_regressors(self):
        ensemble = Ensemble(type_=ENSEMBLE_TYPE_REGRESSORS)
        assert ensemble.type_ == ENSEMBLE_TYPE_REGRESSORS

    def test_ensemble_type_none(self):
        ensemble = Ensemble()
        assert ensemble.type_ is None

    def test_ensemble_len_0(self):
        ensemble = Ensemble()
        assert len(ensemble) == 0

    def test_ensemble_len_1(self):
        ensemble = Ensemble()
        for i in range(10):
            assert len(ensemble) == i
            ensemble.append(DecisionTreeClassifier())

    def test_ensemble_len_2(self):
        ensemble = Ensemble([DecisionTreeRegressor(), DecisionTreeRegressor()])
        assert len(ensemble) == 2

    def test_ensemble_classes(self):
        y1 = np.random.choice([1,2], len(y), replace=True)
        y2 = np.random.choice([3,4], len(y), replace=True)
        dt1 = DecisionTreeClassifier(max_depth=1).fit(X, y1)
        dt2 = DecisionTreeClassifier(max_depth=10).fit(X, y2)

        ensemble = Ensemble([dt1, dt2])
        assert np.array_equal(ensemble.classes_, [1,2,3,4])

    def test_ensemble_output(self):
        y1 = np.random.choice([1,2], len(y), replace=True)
        y2 = np.random.choice([3,4], len(y), replace=True)
        dt1 = DecisionTreeClassifier(max_depth=10).fit(X, y1)
        dt2 = DecisionTreeClassifier(max_depth=10).fit(X, y2)

        ensemble = Ensemble([dt1, dt2])
        output = ensemble.output(X)
        labels = output2labels(output, classes=ensemble.classes_)

        assert np.array_equal(labels[:,0], dt1.predict(X))
        assert np.array_equal(labels[:,1], dt2.predict(X))


def TestOutputs():

    def test_output2votes(self):
        pass

    def test_output2labels(self):
        pass


