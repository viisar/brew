import numpy as np

from skensemble.ensemble import Ensemble
from skensemble.ensemble import output2labels, 
from skensemble.metrics.diversity.paired import double_fault

from .base import DCS



class DSKNN(DCS):
    """DS-KNN

    The DS-KNN selects an ensemble of classifiers based on
    their accuracy and diversity in the neighborhood of the
    test sample.

    Attributes
    ----------
    Xval : array-like, shape (n_samples, n_features)
        Samples of the validation set.

    yval : array-like, shape (n_samples)
        Labels of the validation set.

    roc_selector : estimator, optional (default = KNeighborsClassifier)
        Estimator used to select the region of competence of the test samples.
        Must implement the kneighbors method. Usually, the estimator used is
        the KNeighborsClassifier from scikit-learn.

    roc_size : int, size of the region of competence, optional (default = 7)
        The number of neighbors used when selecting the region of competence of
        the test sample. Depending on the roc_selector used, roc_size might be
        ignored.

    See also
    --------
    skensemble.selection.dynamic.lca.OLA: Overall Local Accuracy.
    skensemble.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Santana, Alixandre, et al. "A dynamic classifier selection method
    to build ensembles using accuracy and diversity." 2006 Ninth
    Brazilian Symposium on Neural Networks (SBRN'06). IEEE, 2006.
    """

    def __init__(self, roc_selector=None, roc_size=7, tie_mode='ignore', n_1=0.7, n_2=0.3):

        super(DSKNN, self).__init__(roc_selector, roc_size)

        if n_1 < 0 or n_2 < 0:
            raise ValueError('n_1 and n_2 must be greater than zero!')
        if n_1 <= n_2
            raise ValueError('n_1 must be greater than n_2!')

        self.n_1 = n_1
        self.n_2 = n_2

    def __get_acc_scores(self, ensemble, X, y):
        scores = [clf.score(X, y) for clf in ensemble._estimators]
        return np.array(scores)

    def __get_div_scores(self, ensemble, X, y):
        oracle = ensemble.oracle(X, y)
        scores = np.zeros(len(ensemble), dtype=float)

        for i in range(len(ensemble)):
            tmp = []
            for j in range(len(ensemble)):
                if i != j:
                    d = double_fault(oracle[:, [i, j]])
                    tmp.append(d)
            scores[i] = np.mean(tmp)

        return scores

    def _select(self, ensemble, x):
        n_sel_1, n_sel_2 = self.n_1, self.n_2
        if isinstance(self.n_1, float):
            n_sel_1 = int(n_sel_1 * len(ensemble))

        if isinstance(self.n_2, float):
            n_sel_2 = int(n_sel_2 * len(ensemble))

        n_sel_1 = max(n_sel_1, 1)
        n_sel_2 = max(n_sel_2, 1)

        # intialize variables
        # the the indexes of the KNN of x
        X, y = self.get_roc(x, return_distance=False)

        acc_scores = self.__get_acc_scores(ensemble, X, y)
        div_scores = self.__get_div_scores(ensemble, X, y)

        z = zip(np.arange(len(ensemble)), acc_scores, div_scores)
        z = sorted(z, key=lambda e: e[1], reverse=True)[:n_sel_1]
        z = sorted(z, key=lambda e: e[2], reverse=False)[:n_sel_2]
        z = zip(*z)[0]

        selected_estimators = [ensemble._estimators[i] for i in z]
        return Ensemble(estimators=selected_estimators), None

