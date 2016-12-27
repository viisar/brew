import numpy as np

from abc import abstractmethod

from sklearn.utils import check_random_state

from skensemble.ensemble import Ensemble
from .base import DCS

class Probabilistic(DCS):
    #TODO handle base classifiers without predict_proba method
    # without requiring classes to follow the LabelEncoder pattern.
    """Base class for APriori and APosteriori dynamic selection.

    WARNING: this class should not be initialized.
    Use either APriori or APosteriori classes.
    """
    def __init__(self, roc_selector=None, roc_size=7, threshold=0.1, random_state=None):
        super(Probabilistic, self).__init__(roc_selector, roc_size)

        if threshold <= 0 or threshold >= 1:
            raise ValueError('threshold must be in the interval (0,1)!')

        self.threshold = threshold
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def probabilities(self, clf, nn_X, nn_y, distances, x):
        pass

    def __proba_scores(self, clf, X):
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(X)
        elif hasattr(clf, 'decision_function'):
            score = clf.decision_function(X)
            if np.ndim(score) == 1:
                d = {class_ : idx for idx, class_ in enumerate(clf.classes_)}
                labels = [d[class_] for class_ in clf.predict(X)]

                new_score = np.zeros((X.shape[0], 2), dtype=float)
                for i in range(len(labels)):
                    new_score[i, labels[i]] = score[i]
                score = np.abs(new_score)

            proba = np.exp(score) / np.sum(np.exp(score), axis=1)[:, np.newaxis]

        return proba


    def _select(self, ensemble, x):
        selected_classifier = None

        X, y, dists = self.get_roc(x, return_distance=True)

        p_correct = np.zeros(len(ensemble), dtype=float)
        for j in range(len(ensemble)):
            p_correct[j] = self.probabilities(clf, X, y, dists, x)

        [idx_selected] = np.where(p_correct >= 0.5)
        if len(idx_selected) == 0:
            idx_selected = [np.argmax(p_correct)]
        
        m = np.argmax(p_correct)
        selected = True

        d = np.zeros(len(idx_selected), dtype=float)
        for i, j in enumerate(idx_selected):
            d[i] = p_correct[m] - p_correct[j]
            if j != m and d[i] < self.threshold:
                selected = False
                # can not break because has to calculate all d[i]

        if selected == True:
            selected_classifier = ensemble._estimators[m]
        else:
            idx_selected = idx_selected[d < self.threshold]
            idx = self.random_state.choice(idx_selected)
            selected_classifier = self._estimators[idx]

        return Ensemble([selected_classifier]), None


class APriori(Probabilistic):
    """A Priori Classifier Selection.

    The A Priori method is a dynamic classifier selection that
    uses a probabilistic-based measures for selecting the best
    classifier.

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

    threshold : float, optional (default = 0.1)
        Threshold used to verify if there is a single best.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    See also
    --------
    skensemble.selection.dynamic.probabilistic.APosteriori: A Posteriori DCS.
    skensemble.selection.dynamic.ola.OLA: Overall Local Accuracy.
    skensemble.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Methods for dynamic classifier
    selection." Image Analysis and Processing, 1999. Proceedings.
    International Conference on. IEEE, 1999.

    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.
    """
    def __init__(self, roc_selector=None, roc_size=7, threshold=0.1, random_state=None):
        super(APriori, self).__init__(roc_selector=roc_selector, roc_size=roc_size, 
                threshold=threshold, random_state=random_state)

    def probabilities(self, clf, X, y, distances, x):
        proba = self.__proba_scores(clf, X)

        d = dict(list(enumerate(clf.classes_)))
        proba_col = [d[y_i] for y_i in y if y_i in d else None]

        probabilities = [proba[i, proba_col[i]] for i in range(len(y))\
                if col[i] is not None else 0.0]
        probabilities = np.array(proba)

        delta = 1. / (distances + 10e-8)

        p_correct = np.sum(probabilities * delta) / np.sum(delta)
        return p_correct


class APosteriori(Probabilistic):
    """A Posteriori Classifier Selection.

    The A Priori method is a dynamic classifier selection that
    uses a probabilistic-based measures for selecting the best
    classifier.

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

    threshold : float, optional (default = 0.1)
        Threshold used to verify if there is a single best.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    See also
    --------
    skensemble.selection.dynamic.probabilistic.APriori: A Priori DCS.
    skensemble.selection.dynamic.ola.OLA: Overall Local Accuracy.
    skensemble.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Methods for dynamic classifier
    selection." Image Analysis and Processing, 1999. Proceedings.
    International Conference on. IEEE, 1999.

    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.
    """
    def __init__(self, roc_selector=None, roc_size=7, threshold=0.1, random_state=None):
        super(APosteriori, self).__init__(roc_selector=roc_selector,
                roc_size=roc_size, threshold=threshold,
                random_state=random_state)

    def probabilities(self, clf, X, y, distances, x):
        [w_l] = clf.predict(x.reshape(1,-1))
        [idx_w_l] = np.where(y == w_l)
        [proba_col] = np.where(clf.classes_ == w_l)

        proba = self.__proba_scores(clf, X)

        delta = 1. / (distances + 10e-8)

        numerator = np.sum(proba[idx_w_l, proba_col].ravel() * delta[idx_w_l])
        denominator = np.sum(proba[:, proba_col].ravel() * delta)

        # if the classifier never classifies as class w_l, P(w_l|psi_i) = 0
        return float(numerator) / (denominator + 10e-8)

