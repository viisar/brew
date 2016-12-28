import numpy as np

from skensemble.ensemble import Ensemble
from skensemble.ensemble import output2labels

from .base import DCS


class MCB(DCS):
    """Multiple Classifier Behavior.

    The Multiple Classifier Behavior (MCB) selects the best
    classifier using the similarity of the classifications
    on the K neighbors of the test sample in the validation
    set.

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

    similarity_threshold : float, optional (default = 0.7)
        The similarity threshold used to select a subset of samples from
        the region of competence to be used when evaluating the competence
        of base classifiers.

    significance_threshold : float, optional (default = 0.3)
        The significance threshold used to decide if the single most competent
        classifier in the pool will be selected, or if the whole ensemble of
        classifiers will be selected.

    See also
    --------
    skensemble.selection.dynamic.ola.OLA: Overall Local Accuracy.
    skensemble.selection.dynamic.lca.LCA: Local Class Accuracy.

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Dynamic classifier selection
    based on multiple classifier behaviour." Pattern Recognition 34.9
    (2001): 1879-1881.
    """

    def __init__(self, roc_selector=None, roc_size=7, similarity_threshold=0.7, significance_threshold=0.3):
        super(MCB, self).__init__(roc_selector, roc_size)

        if similarity_threshold != np.clip(similarity_threshold, 0, 1):
            raise ValueError('similarity_threshold must be in the interval [0,1]')

        if significance_threshold != np.clip(significance_threshold,0,1):
            raise ValueError('significance_threshold must be in the interval [0,1]')

        self.similarity_threshold = similarity_threshold
        self.significance_threshold = significance_threshold

    def _select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]]), None

        out_x = ensemble.output(x.reshape(1,-1))
        mcb_x = output2labels(out_x, ensemble.classes_).ravel()

        # intialize variables
        # the the indexes of the KNN of x
        X, y = self.get_roc(x, return_distance=False)
        out_v = ensemble.output(X)
        mcb_v = output2labels(out_v, classes=ensemble.classes_)

        # compute the similarity between the pool output
        # for the test sample and for the sample in X (ROC).
        # if the similarity is greater than similarity_threshold
        # add this sample to ROC' (new ROC).
        idx = []
        for i in range(X.shape[0]):
            sim = np.mean(mcb_x == mcb_v[i, :])
            if sim > self.similarity_threshold:
                idx = idx + [i]

        if len(idx) == 0:
            idx = np.arange(X.shape[0])
        else:
            idx = np.array(idx)

        X, y = X[idx,:], y[idx]

        scores = np.array([clf.score(X, y) for clf in ensemble._estimators])
        scores = np.array(scores)

        # if best classifier is significantly better
        # use best_classifier
        best_i = np.argmax(scores)
        best_j_score = np.max(scores[np.arange(len(scores)) != best_i])
        if scores[best_i] - best_j_score >= self.significance_threshold:
            best_classifier = ensemble._estimators[best_i]
            return Ensemble(estimators=[best_classifier]), None

        return Ensemble(estimators=ensemble._estimators), None
