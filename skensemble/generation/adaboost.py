import numpy as np

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import is_classifier, is_regressor
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from skensemble import Ensemble
from skensemble.generation.base import BaseEnsembleGenerator

class AdaBoost(BaseEnsembleGenerator):
    """AdaBoost Ensemble Generator.

    Adaboost is an ensemble generation method that begins by
    fitting an estimator on the original dataset and then fits
    additional copies of the estimator on the same dataset but
    where the weights of difficult samples are adjusted such 
    that subsequent estimators can focus on those samples.
        
    Parameters
    ----------
    base_estimator : object, estimator
        Base estimator used to build the ensemble.

    n_estimators : int, optional (default=100)
        Maximum number of base estimators in the ensemble.
        Stops when it reaches 100\% accuracy or number of estimators
        reaches n_estimators.

    learning_rate : float, optional (default=1.0)
        Learning rate shrinks the contribution of each estimator by learning_rate.
        There is a trade-off between learning_rate and n_estimators.

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
    Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of on-Line
    Learning iand an Application to Boosting", 1995.

    J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
    """
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
 
