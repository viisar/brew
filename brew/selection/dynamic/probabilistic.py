import numpy as np
import abc

from brew.base import Ensemble
from .base import DCS


class Probabilistic(DCS):

    def __init__(self,
                 Xval,
                 yval,
                 K=5,
                 weighted=False,
                 knn=None,
                 threshold=0.1):
        self.threshold = threshold
        super(Probabilistic, self).__init__(
            Xval, yval, K=K, weighted=weighted, knn=knn)

    @abc.abstractmethod
    def probabilities(self, clf, nn_X, nn_y, distances, x):
        pass

    def select(self, ensemble, x):
        selected_classifier = None

        nn_X, nn_y, dists = self.get_neighbors(x,
                                               return_distance=True)

        idx_selected, prob_selected = [], []

        all_probs = np.zeros(len(ensemble))
        for idx, clf in enumerate(ensemble.classifiers):
            prob = self.probabilities(clf, nn_X, nn_y, dists, x)
            if prob > 0.5:
                idx_selected = idx_selected + [idx]
                prob_selected = prob_selected + [prob]

            all_probs[idx] = prob

        if len(prob_selected) == 0:
            prob_selected = [np.max(all_probs)]
            idx_selected = [np.argmax(all_probs)]

        p_correct_m = max(prob_selected)
        m = np.argmax(prob_selected)

        selected = True
        diffs = []
        for j, p_correct_j in enumerate(prob_selected):
            d = p_correct_m - p_correct_j
            diffs.append(d)
            if j != m and d < self.threshold:
                selected = False

        if selected:
            selected_classifier = ensemble.classifiers[idx_selected[m]]
        else:
            idx_selected = np.asarray(idx_selected)
            mask = np.array(np.array(diffs) < self.threshold, dtype=bool)
            i = np.random.choice(idx_selected[mask])
            selected_classifier = ensemble.classifiers[i]

        return Ensemble([selected_classifier]), None


class APriori(Probabilistic):
    """A Priori Classifier Selection.

    The A Priori method is a dynamic classifier selection that
    uses a probabilistic-based measures for selecting the best
    classifier.

    Attributes
    ----------
    `Xval` : array-like, shape = [indeterminated, n_features]
        Validation set.

    `yval` : array-like, shape = [indeterminated]
        Labels of the validation set.

    `knn`  : sklearn KNeighborsClassifier,
        Classifier used to find neighborhood.

    `threshold`  : float, default = 0.1
        Threshold used to verify if there is a single best.

    Examples
    --------
    >>> from brew.selection.dynamic.probabilistic import APriori
    >>> from brew.generation.bagging import Bagging
    >>> from brew.base import EnsembleClassifier
    >>>
    >>> from sklearn.tree import DecisionTreeClassifier as Tree
    >>> import numpy as np
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0] ,
                      [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> tree = Tree(max_depth=1, min_samples_leaf=1)
    >>> bag = Bagging(base_classifier=tree, n_classifiers=10)
    >>> bag.fit(X, y)
    >>>
    >>> apriori = APriori(X, y, K=3)
    >>>
    >>> clf = EnsembleClassifier(bag.ensemble, selector=apriori)
    >>> clf.predict([-1.1,-0.5])
    [1]

    See also
    --------
    brew.selection.dynamic.probabilistic.APosteriori: A Posteriori DCS.
    brew.selection.dynamic.ola.OLA: Overall Local Accuracy.
    brew.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Methods for dynamic classifier
    selection." Image Analysis and Processing, 1999. Proceedings.
    International Conference on. IEEE, 1999.

    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.
    """

    def __init__(self,
                 Xval,
                 yval,
                 K=5,
                 weighted=False,
                 knn=None,
                 threshold=0.1):
        self.threshold = threshold
        super(APriori, self).__init__(
            Xval, yval, K=K, weighted=weighted, knn=knn)

    def probabilities(self, clf, nn_X, nn_y, distances, x):
        # in the A Priori method, the 'x' is not used
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(nn_X)
        elif hasattr(clf, 'decision_function'):
            dc = clf.decision_function(nn_X)
            if len(dc.shape) == 1:
                cl = clf.predict(nn_X).astype(int)
                sc = np.zeros((dc.shape[0], 2))
                for i in range(len(nn_X)):
                    sc[i, cl[i]] = dc[i]
                dc = np.abs(sc)
            proba = np.exp(dc) / np.sum(np.exp(dc), axis=1)[:, np.newaxis]

        d = dict(list(enumerate(clf.classes_)))
        col_idx = np.zeros(nn_y.size, dtype=int)
        for i in range(nn_y.size):
            col_idx[i] = d[nn_y[i]] if nn_y[i] in d else proba.shape[1] - 1

        probabilities = proba[np.arange(col_idx.size), col_idx]
        delta = 1. / (distances + 10e-8)

        p_correct = np.sum(probabilities * delta) / np.sum(delta)
        return p_correct


class APosteriori(Probabilistic):
    """A Priori Classifier Selection.

    The A Priori method is a dynamic classifier selection that
    uses a probabilistic-based measures for selecting the best
    classifier.

    Attributes
    ----------
    `Xval` : array-like, shape = [indeterminated, n_features]
        Validation set.

    `yval` : array-like, shape = [indeterminated]
        Labels of the validation set.

    `knn`  : sklearn KNeighborsClassifier,
        Classifier used to find neighborhood.

    `threshold`  : float, default = 0.1
        Threshold used to verify if there is a single best.

    Examples
    --------
    >>> from brew.selection.dynamic.probabilistic import APosteriori
    >>> from brew.generation.bagging import Bagging
    >>> from brew.base import EnsembleClassifier
    >>>
    >>> from sklearn.tree import DecisionTreeClassifier as Tree
    >>> import numpy as np
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1],
                      [-0.5, 0] , [0.5, 0], [1, 0],
                      [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> tree = Tree(max_depth=1, min_samples_leaf=1)
    >>> bag = Bagging(base_classifier=tree, n_classifiers=10)
    >>> bag.fit(X, y)
    >>>
    >>> aposteriori = APosteriori(X, y, K=3)
    >>>
    >>> clf = EnsembleClassifier(bag.ensemble, selector=aposteriori)
    >>> clf.predict([-1.1,-0.5])
    [1]

    See also
    --------
    brew.selection.dynamic.probabilistic.APriori: A Priori DCS.
    brew.selection.dynamic.ola.OLA: Overall Local Accuracy.
    brew.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Methods for dynamic classifier
    selection." Image Analysis and Processing, 1999. Proceedings.
    International Conference on. IEEE, 1999.

    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.
    """

    def __init__(self,
                 Xval,
                 yval,
                 K=5,
                 weighted=False,
                 knn=None,
                 threshold=0.1):
        self.threshold = threshold
        super(APosteriori, self).__init__(
            Xval, yval, K=K, weighted=weighted, knn=knn)

    def probabilities(self, clf, nn_X, nn_y, distances, x):
        [w_l] = clf.predict(x)
        [idx_w_l] = np.where(nn_y == w_l)
        [proba_col] = np.where(clf.classes_ == w_l)

        # in the A Posteriori method the 'x' is used
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(nn_X)
        elif hasattr(clf, 'decision_function'):
            dc = clf.decision_function(nn_X)
            if len(dc.shape) == 1:
                cl = clf.predict(nn_X).astype(int)
                sc = np.zeros((dc.shape[0], 2))
                for i in range(len(nn_X)):
                    sc[i, cl[i]] = dc[i]
                dc = np.abs(sc)
            proba = np.exp(dc) / np.sum(np.exp(dc), axis=1)[:, np.newaxis]

        # if the classifier never classifies as class w_l, P(w_l|psi_i) = 0

        delta = 1. / (distances + 10e-8)

        numerator = sum(proba[idx_w_l, proba_col].ravel() * delta[idx_w_l])
        denominator = sum(proba[:, proba_col].ravel() * delta)
        return float(numerator) / (denominator + 10e-8)
