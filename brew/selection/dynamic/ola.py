import numpy as np

from brew.base import Ensemble
from .base import DCS


class OLA2(DCS):
    """Overall Local Accuracy.

    The Overall Local Accuracy selects the best classifier for
    a sample using it's K nearest neighbors.

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
    >>> from brew.selection.dynamic.ola import OLA
    >>> from brew.generation.bagging import Bagging
    >>> from brew.base import EnsembleClassifier
    >>>
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> import numpy as np
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0], [0.5, 0],
                      [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> tree = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    >>> bag = Bagging(base_classifier=tree, n_classifiers=10)
    >>> bag.fit(X, y)
    >>>
    >>> ola = OLA(X, y, K=3)
    >>>
    >>> clf = EnsembleClassifier(bag.ensemble, selector=ola)
    >>> clf.predict([-1.1,-0.5])
    [1]

    See also
    --------
    brew.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Woods, Kevin, Kevin Bowyer, and W. Philip Kegelmeyer Jr. "Combination
    of multiple classifiers using local accuracy estimates." Computer Vision
    and Pattern Recognition, 1996. Proceedings CVPR'96, 1996 IEEE Computer
    Society Conference on. IEEE, 1996.

    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.
    """

    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]]), None

        # intialize variables
        # the the indexes of the KNN of x
        classifiers = ensemble.classifiers
        [idx] = self.knn.kneighbors(x, return_distance=False)
        X, y = self.Xval[idx], self.yval[idx]

        # d[score] = indexes of the classifiers with that score
        d = {}
        scores = [clf.score(X, y) for clf in ensemble.classifiers]
        for i, scr in enumerate(scores):
            d[scr] = d[scr] + [i] if scr in d else [i]
        best_scores = sorted([k for k in list(d.keys())], reverse=True)

        # if there was a single best classifier, return it
        if len(d[best_scores[0]]) == 1:
            i = d[best_scores[0]][0]
            return Ensemble([classifiers[i]]), None

        options = None
        for j, score in enumerate(best_scores):
            pred = [classifiers[index].predict(x) for index in d[score]]
            pred = np.asarray(pred).flatten()

            bincount = np.bincount(pred.astype(int))
            if options is not None:
                for i in range(len(bincount)):
                    bincount[i] = bincount[i] if i in options else 0

            imx = np.argmax(bincount)
            votes = np.argwhere(bincount == bincount[imx]).flatten()
            count = len(votes)
            if count == 1:
                return Ensemble([classifiers[np.argmax(pred == imx)]]), None
            elif options is None:
                options = votes

        return Ensemble([classifiers[np.argmax(scores)]]), None


class OLA(DCS):

    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]]), None

        # intialize variables
        # the the indexes of the KNN of x
        classifiers = ensemble.classifiers
        [idx] = self.knn.kneighbors(x, return_distance=False)
        X, y = self.Xval[idx], self.yval[idx]

        scores = np.asarray([clf.score(X, y) for clf in classifiers])

        return Ensemble([classifiers[np.argmax(scores)]]), None
