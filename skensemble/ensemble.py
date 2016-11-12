import numpy as np

import sklearn.base

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
    def __init__(self, estimators=None, is_classification=True):
        if estimators is None:
            self._estimators = []
        else:
            self._estimators = estimators

        self.is_classification = is_classification

    def __len__(self):
        return len(self._estimators)

    @property
    def classes_(self):
        classes = set()
        
        for clf in self._estimators:
            if clf.classes_ is not None:
                classes = classes.union(set(clf.classes_))

        return np.array(list(classes))

    def add(self, estimator):
        self._estimators.append(estimator)


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
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_estimators = len(self._estimators)

        if self.is_classification:
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
        if not self.is_classification:
            raise NotImplementedError
        pass

    def oracle(self, X, y):
        if not self.is_classification:
            raise NotImplementedError

def output2votes(output):
    votes = np.zeros_like(output, dtype=int)

    for idx_estimator in range(votes.shape[2]):
        idx_classes = np.argmax(output[:, :, idx_estimator], axis=1)
        votes[np.arange(votes.shape[0]), idx_classes, idx_estimator] = 1

    return votes

