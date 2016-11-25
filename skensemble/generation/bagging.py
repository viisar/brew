import numpy as np

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import clone, is_classifier, is_regressor

from skensemble import Ensemble
from skensemble.generation.base import BaseEnsembleGenerator

class Bagging(BaseEnsembleGenerator):

    def __init__(self, base_estimator, n_estimators=100, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = check_random_state(random_state)
        self._ensemble = None

        if n_estimators < 1:
            raise ValueError('n_estimators must be greater than 0!')

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self._ensemble = Ensemble()

        for i in range(self.n_estimators):
            idx = self.random_state.choice(X.shape[0], X.shape[0], replace=True)
            X_, y_ = X[idx, :], y[idx]
            estimator = clone(self.base_estimator)
            self._ensemble.add(estimator.fit(X_, y_))

        return self

    @property
    def ensemble(self):
        if hasattr(self, '_ensemble') and self._ensemble is not None:
            return self._ensemble
        else:
            raise Exception('Must fit before get ensemble!')

class BaggingClassification(BaseEnsembleGenerator):

    def __init__(self, base_estimator, n_estimators=100, random_state=None):
        if not is_classifier(base_estimator):
            raise ValueError('base_estimator must be a classifier!')

        super(BaggingClassification, self).__init__(base_estimator, 
                n_estimators, random_state)

    def fit(self, X, y):
        super(BaggingClassification, self).fit(X, y)
        return self

class BaggingRegression(BaseEnsembleGenerator):

    def __init__(self, base_estimator, n_estimators=100, random_state=None):
        if not is_regressor(base_estimator):
            raise ValueError('base_estimator must be a regressor!')

        super(BaggingRegression, self).__init__(base_estimator, 
                n_estimators, random_state)

    def fit(self, X, y):
        super(BaggingRegression, self).fit(X, y)
        return self

