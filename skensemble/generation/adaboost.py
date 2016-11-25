import numpy as np

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import is_classifier, is_regressor
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from skensemble import Ensemble
from skensemble.generation.base import BaseEnsembleGenerator

class AdaBoost(BaseEnsembleGenerator):

    def __init__(self, base_estimator, n_estimators=100, learning_rate=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self.random_state = check_random_state(random_state)
        self._ensemble = None

        if n_estimators < 1:
            raise ValueError('n_estimators must be greater than 0!')

        if is_classifier(self.base_estimator):
            if hasattr(self.base_estimator, 'predict_proba'):
                algorithm = 'SAMME.R'
            else:
                algorithm = 'SAMME'

            self.gen = AdaBoostClassifier(base_estimator, n_estimators=n_estimators,
                    learning_rate=learning_rate, algorithm=algorithm)
        elif is_regressor(self.base_estimator):
            self.gen = AdaBoostRegressor(base_estimator, n_estimators=n_estimators,
                    learning_rate=learning_rate)
        else:
            raise ValueError('Invalid base_estimator!')

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self._ensemble = Ensemble()
        estimators_ = self.gen.fit(X, y).estimators_
        #for estimator in estimators_:
        #    estimator.classes_ = self.gen.classes_.take(estimator.classes_)
        self._ensemble = Ensemble(estimators=estimators_)

    @property
    def ensemble(self):
        if hasattr(self, '_ensemble') and self._ensemble is not None:
            return self._ensemble
        else:
            raise Exception('Must fit before get ensemble!')
 
