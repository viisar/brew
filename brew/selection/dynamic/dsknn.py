import numpy as np

from brew.base import Ensemble
from brew.metrics.diversity.paired import kuncheva_double_fault_measure
from .base import DCS


class DSKNN(DCS):
    """DS-KNN

    The DS-KNN selects an ensemble of classifiers based on
    their accuracy and diversity in the neighborhood of the
    test sample.

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
    >>> from brew.selection.dynamic import DSKNN
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
    >>> sel = DSKNN(X, y, K=3)
    >>>
    >>> clf = EnsembleClassifier(bag.ensemble, selector=sel)
    >>> clf.predict([-1.1,-0.5])
    [1]

    See also
    --------
    brew.selection.dynamic.lca.OLA: Overall Local Accuracy.
    brew.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Santana, Alixandre, et al. "A dynamic classifier selection method
    to build ensembles using accuracy and diversity." 2006 Ninth
    Brazilian Symposium on Neural Networks (SBRN'06). IEEE, 2006.
    """

    def __init__(self, Xval, yval, K=5, weighted=False, knn=None,
                 n_1=0.7, n_2=0.3):
        if n_1 < 0 or n_2 < 0 or n_1 <= n_2:
            raise Exception

        self.n_1 = n_1
        self.n_2 = n_2
        super(DSKNN, self).__init__(
            Xval, yval, K=K, weighted=weighted, knn=knn)

    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]]), None

        n_sel_1, n_sel_2 = self.n_1, self.n_2
        if isinstance(self.n_1, float):
            n_sel_1 = int(n_sel_1 * len(ensemble))

        if isinstance(self.n_2, float):
            n_sel_2 = int(n_sel_2 * len(ensemble))

        n_sel_1 = max(n_sel_1, 1)
        n_sel_2 = max(n_sel_2, 1)

        # intialize variables
        # the the indexes of the KNN of x
        classifiers = ensemble.classifiers
        [idx] = self.knn.kneighbors(x, return_distance=False)
        X, y = self.Xval[idx], self.yval[idx]

        acc_scores = np.array([clf.score(X, y) for clf in classifiers])

        out = ensemble.output(X, mode='labels')
        oracle = np.equal(out, y[:, np.newaxis])
        div_scores = np.zeros(len(ensemble), dtype=float)

        for i in range(len(ensemble)):
            tmp = []
            for j in range(len(ensemble)):
                if i != j:
                    d = kuncheva_double_fault_measure(oracle[:, [i, j]])
                    tmp.append(d)
            div_scores[i] = np.mean(tmp)

        z = zip(np.arange(len(ensemble)), acc_scores, div_scores)
        z = sorted(z, key=lambda e: e[1], reverse=True)[:n_sel_1]
        z = sorted(z, key=lambda e: e[2], reverse=False)[:n_sel_2]
        z = zip(*z)[0]

        classifiers = [classifiers[i] for i in z]
        return Ensemble(classifiers=classifiers), None
