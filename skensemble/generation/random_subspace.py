import numpy as np

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import is_classifier, is_regressor
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from skensemble import Ensemble
from skensemble.generation.base import BaseEnsembleGenerator

class _RandomSubspaceTransformer(TransformerMixin):
    
    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        if self.features is None:
            return X

        return X[:, self.features]

class RandomSubspace(BaseEnsembleGenerator):

    def __init__(self, base_estimator, n_estimators=100, max_features=0.5, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_features = max_features

        self.estimators_features = None

        self.random_state = check_random_state(random_state)
        self._ensemble = None

        if n_estimators < 1:
            raise ValueError('n_estimators must be greater than 0!')

        if max_features <= 0.0:
            raise ValueError('max_features must be greater than 0!')


    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.estimators_features = []

        self._ensemble = Ensemble()

        if isinstance(self.max_features, float):
            max_features = int(max(self.max_features * X.shape[1], 1))
        else:
            max_features = self.max_features

        for i in range(self.n_estimators):
            n_features = self.random_state.randint(1, max_features + 1)
            chosen_features = self.random_state.choice(X.shape[1], n_features)
            pipe = Pipeline([
                    ('transformer', _RandomSubspaceTransformer(chosen_features)),
                    ('estimator', clone(self.base_estimator))])

            self._ensemble.add(pipe.fit(X, y))
            self.estimators_features.append(chosen_features)

        return self

    @property
    def ensemble(self):
        if hasattr(self, '_ensemble') and self._ensemble is not None:
            return self._ensemble
        else:
            raise Exception('Must fit before get ensemble!')
 
