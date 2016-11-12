import numpy as np

from brew.base import Ensemble
from .base import DCS


class MCB(DCS):
    """Multiple Classifier Behavior.

    The Multiple Classifier Behavior (MCB) selects the best
    classifier using the similarity of the classifications
    on the K neighbors of the test sample in the validation
    set.

    Attributes
    ----------
    `Xval` : array-like, shape = [indeterminated, n_features]
        Validation set.

    `yval` : array-like, shape = [indeterminated]
        Labels of the validation set.

    `knn`  : sklearn KNeighborsClassifier,
        Classifier used to find neighborhood.


    Examples
    --------
    >>> from brew.selection.dynamic.mcb import MCB
    >>> from brew.generation.bagging import Bagging
    >>> from brew.base import EnsembleClassifier
    >>>
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> import numpy as np
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0],
                      [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> tree = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    >>> bag = Bagging(base_classifier=tree, n_classifiers=10)
    >>> bag.fit(X, y)
    >>>
    >>> mcb = MCB(X, y, K=3)
    >>>
    >>> clf = EnsembleClassifier(bag.ensemble, selector=mcb)
    >>> clf.predict([-1.1,-0.5])
    [1]

    See also
    --------
    brew.selection.dynamic.lca.OLA: Overall Local Accuracy.
    brew.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Dynamic classifier selection
    based on multiple classifier behaviour." Pattern Recognition 34.9
    (2001): 1879-1881.
    """

    def __init__(self, Xval, yval, K=5, weighted=False, knn=None,
                 similarity_threshold=0.7, significance_threshold=0.3):
        self.similarity_threshold = similarity_threshold
        self.significance_threshold = significance_threshold
        super(MCB, self).__init__(Xval, yval, K=K, weighted=weighted, knn=knn)

    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]]), None

        mcb_x = ensemble.output(x, mode='labels')[0, :]

        # intialize variables
        # the the indexes of the KNN of x
        [idx] = self.knn.kneighbors(x, return_distance=False)
        X, y = self.Xval[idx], self.yval[idx]
        mcb_v = ensemble.output(X, mode='labels')

        idx = []
        for i in range(X.shape[0]):
            sim = np.mean(mcb_x == mcb_v[i, :])
            if sim > self.similarity_threshold:
                idx = idx + [i]

        if len(idx) == 0:
            idx = np.arange(X.shape[0])

        scores = [clf.score(X[idx], y[idx]) for clf in ensemble.classifiers]
        scores = np.array(scores)

        # if best classifier is significantly better
        # use best_classifier
        best_i = np.argmax(scores)
        best_j_score = np.max(scores[np.arange(len(scores)) != best_i])
        if scores[best_i] - scores[best_j] >= self.significance_threshold:
            best_classifier = ensemble.classifiers[best_i]
            return Ensemble(classifiers=[best_classifier]), None

        return Ensemble(classifiers=ensemble.classifiers), None
