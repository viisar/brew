import numpy as np

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import is_classifier, is_regressor
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from skensemble import Ensemble
from skensemble.generation.base import BaseEnsembleGenerator

class _RandomSubspaceTransformer(TransformerMixin):
    """Random Subspace Transformer.

    Transformer used to select the subset of features
    before training and predicting the classes of samples
    in the training set.

    Parameters
    ----------
    features : ndarray, list of indexes
        List of indexes of features to be selected.

    Attributes
    ----------
    features_ : ndarray, list of indexes
        List of indexes of features to be selected.
    """
    def __init__(self, features=None):
        self.features_ = features

    def fit(self, X, y=None, **fit_params):
        """Method added for sklearn Pipeline compatibility.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """Apply dimensionality reduction on X.

        Selects a subset of features from X.

        Parameters
        ----------
        X : ndarray, shape = (n_samples, n_features)
            The input data to be transformed.

        Returns
        -------
        X_new : array-like, shape (n_samples, features_)
            transformed array where features_ is a subset
            of features from X.
        """
        if self.features_ is None:
            return X

        return X[:, self.features_]

class RandomSubspace(BaseEnsembleGenerator):
    """A Random Subspace Ensemble Generator.

    Random Subspace is an ensemble generation method that
    fits base estimators (classifiers or regressors) using
    random subsets of features from the training set.
        
    Parameters
    ----------
    base_estimator : object, estimator
        Base estimator used to build the ensemble.

    n_estimators : int, optional (default=100)
        Number of base estimators in the ensemble

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

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
    T. Ho, "The random subspace method for constructing decision forests",
    Pattern Analysis and Machine Intelligence, 20(8), 832-844, 1998.
    """
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

        self.estimators_features = []

        self._ensemble = Ensemble()

        if isinstance(self.max_features, float):
            max_features = int(max(self.max_features * X.shape[1], 1))
        else:
            max_features = self.max_features

        for i in range(self.n_estimators):
            n_features = self.random_state.randint(1, max_features + 1)
            chosen_features = self.random_state.choice(X.shape[1], n_features,
                    replace=False)
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
 
