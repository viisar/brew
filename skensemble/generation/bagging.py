import numpy as np

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import clone, is_classifier, is_regressor

from skensemble import Ensemble
from skensemble.generation.base import BaseEnsembleGenerator

class Bagging(BaseEnsembleGenerator):
    """A Bagging Ensemble Generator.

    Bagging is an ensemble generation method that fits base
    estimators (classifiers or regressors) using random subsets
    of samples from the training set.
        
    Parameters
    ----------
    base_estimator : object, estimator
        Base estimator used to build the ensemble.

    n_estimators : int, optional (default=100)
        Number of base estimators in the ensemble.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    base_estimator : estimator
        The base estimator used to build the ensemble.

    ensemble : object Ensemble
        Fitted ensemble of fitted base estimators.

    References
    ----------
    L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140, 1996.
    """
    def __init__(self, base_estimator, n_estimators=100, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = check_random_state(random_state)
        self._ensemble = None

        if n_estimators < 1:
            raise ValueError('n_estimators must be greater than 0!')

    def fit(self, X, y):
        """Build an ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : ndarray, shape = (n_samples, n_features)
            The input data from the training set.
        y : ndarray, shape = (n_samples)
            The target data from the training set.

        Returns
        -------
        self : object
            Returns self.
        """
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

