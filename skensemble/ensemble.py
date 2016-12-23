import numpy as np

import sklearn.utils
from sklearn.base import is_classifier, is_regressor

ENSEMBLE_TYPE_CLASSIFIERS = 'classifiers'
ENSEMBLE_TYPE_REGRESSORS = 'regressors'
ENSEMBLE_TYPE_UNDEFINED = 'undefined'

ENSEMBLE_TYPES = [None, ENSEMBLE_TYPE_CLASSIFIERS, ENSEMBLE_TYPE_REGRESSORS]

def is_ensemble_of_classifiers(ensemble):
    return ensemble.type_ == ENSEMBLE_TYPE_CLASSIFIERS

def is_ensemble_of_regressors(ensemble):
    return ensemble.type_ == ENSEMBLE_TYPE_REGRESSORS

def output2votes(ensemble_output):
    votes = np.zeros_like(output, dtype=int)

    for idx_estimator in range(votes.shape[2]):
        idx_classes = np.argmax(ensemble_output[:, :, idx_estimator], axis=1)
        votes[np.arange(votes.shape[0]), idx_classes, idx_estimator] = 1

    return votes

def output2labels(ensemble_output, classes=None):
    labels = np.argmax(ensemble_output, axis=1)
    if classes is None:
        return labels

    return classes.take(labels)


class Ensemble(object):
    """Class that represents a list of estimators.

    The Ensemble class serves as a wrapper for a list of estimators,
    besides providing a simple way to calculate the output of all the
    estimators in the ensemble.

    Parameters
    ----------
    estimators : list
        list of estimators in the ensemble.

    Attributes
    ----------
    estimators : list
        Stores all estimators in the ensemble.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> from skensemble.base import Ensemble
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0],
                      [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>>
    >>> dt1 = DecisionTreeClassifier()
    >>> dt2 = DecisionTreeClassifier()
    >>>
    >>> dt1.fit(X, y)
    >>> dt2.fit(X, y)
    >>>
    >>> ens = Ensemble(estimators=[dt1, dt2])

    """
    def __init__(self, estimators=None, type_=None):
        self._estimators = []

        if type_ in ENSEMBLE_TYPES:
            self.__type = type_
        else:
            raise ValueError('Invalid type_ \'{}\'! type_ must be'
                    '\'{}\' or \'{}\''.format(type_,
                        ENSEMBLE_TYPE_CLASSIFIERS,
                        ENSEMBLE_TYPE_REGRESSORS))

        if estimators is not None:
            for e in estimators:
                self.append(e)


    def __len__(self):
        return len(self._estimators)

    @property
    def classes_(self):
        classes = set()
        
        for clf in self._estimators:
            if clf.classes_ is not None:
                classes = classes.union(set(clf.classes_))

        return np.array(list(classes))

    @property
    def type_(self):
        if self.__type is not None:
            return self.__type

        if len(self) == 0:
            return None
        elif is_classifier(self._estimators[0]):
            self.__type = ENSEMBLE_TYPE_CLASSIFIERS
            return self.__type
        elif is_regressor(self._estimators[0]):
            self.__type = ENSEMBLE_TYPE_REGRESSORS
            return self.__type

        return ENSEMBLE_TYPE_UNDEFINED


    def __can_append_estimator(self, estimator):
        ensemble_type = self.type_

        if ensemble_type is None or ensemble_type == ENSEMBLE_TYPE_UNDEFINED:
            return is_classifier(estimator) or is_regressor(estimator)

        if ensemble_type == ENSEMBLE_TYPE_CLASSIFIERS:
            return is_classifier(estimator)

        if ensemble_type == ENSEMBLE_TYPE_REGRESSORS:
            return is_regressor(estimator)

        return False


    def append(self, estimator):
        if not self.__can_append_estimator(estimator):
            raise ValueError('Ensemble objects can not mix classifiers'
                    'and regressors!')

        self._estimators.append(estimator)

        if self.type_ is None or self.type_ == ENSEMBLE_TYPE_UNDEFINED:
            if is_classifier(estimator):
                self.__type = ENSEMBLE_TYPE_CLASSIFIERS
            else:
                self.__type = ENSEMBLE_TYPE_REGRESSORS

    def extend(self, estimators):
        for estimator in estimators:
            if not self.__can_append_estimator(estimator):
                raise ValueError('Ensemble objects can not mix classifiers'
                        'and regressors!')

        for estimator in estimators:
            self.append(estimator)

    def output(self, X):
        """
        Returns the output of all classifiers packed in a numpy array.

        This method calculates the output of each estimator, and stores
        them in a array-like shape. The specific shape and the meaning of
        each element depends on the type of estimators.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        Returns
        -------
        output : ndarray, 
            Matrix containing the probabilities or predicted values for X.
            If ensemble of classifiers, output shape is (n_samples, n_classes, n_estimators)
            If ensemble of regressors, output shape is (n_samples, n_estimators)
        """
        X = sklearn.utils.check_array(X)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_estimators = len(self._estimators)

        if self.type_ == ENSEMBLE_TYPE_CLASSIFIERS:
            output = np.zeros((n_samples, n_classes, n_estimators))

            classes_map = {c : i for i, c in enumerate(self.classes_)}

            for idx, estimator in enumerate(self._estimators):
                cols = [classes_map[c] for c in estimator.classes_]
                output[:, cols, idx] = estimator.predict_proba(X)

        else:
            output = np.zeros((n_samples, n_estimators))

            for idx, estimator in enumerate(self._estimators):
                output[:, idx] = estimator.predict(X)

        return output

    def agrees(self, X):
        X = sklearn.utils.check_array(X)
        preds = np.array([clf.predict(X) for clf in self._estimators]).T
        return np.all(np.equal(preds[:,0, np.newaxis], preds), axis=1)

    def oracle(self, X, y):
        X, y = sklearn.utils.check_X_y(X, y)
        labels = output2labels(self.output(X), self.classes_)
        return np.equal(labels, y[:, np.newaxis])

    def fit(self, X, y):
        X, y = sklearn.utils.check_X_y(X, y)
        for estimator in self._estimators:
            estimator.fit(X, y)

        return self

