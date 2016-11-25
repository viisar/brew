import numpy as np

from skensemble import Ensemble

class BaseEnsembleGenerator(object):

    def __init__(self):
        self._ensemble = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class EnsembleGenerator(BaseEnsembleGenerator):
    """WIP"""

    def __init__(self, method, base_estimator, n_estimators=100):
        self.method = method
        self.base_estimator = base_estimator
        self.random_state = check_random_state(random_state)
        raise NotImplementedError

    def fit(self, X, y):
        pass

