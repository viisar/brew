# -*- coding: utf-8 -*-

import numpy as np

from .base import DCS

from skensemble.ensemble import Ensemble

class KNORA_ELIMINATE(DCS):
    """K-nearest-oracles Eliminate.

    The KNORA Eliminate reduces the neighborhood until finds an
    ensemble of classifiers that correctly classify all neighbors.

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
    skensemble.selection.dynamic.knora.KNORA_UNION: KNORA Union.

    References
    ----------
    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira.
    "Dynamic selection of classifiers—A comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.
    """
    def __init__(self, roc_selector=None, roc_size=7):
        super(KNORA_ELIMINATE, self).__init__(roc_selector, roc_size)

    def _select(self, ensemble, x):
        # get the region of competence (ROC).
        X, y = self.get_roc(x, return_distance=False)

        selected_idx = None

        # oracle shape (n_samples, n_estimators)
        oracle = ensemble.oracle(X, y)

        # reduces the roc_size until selects at least 1 classifier
        # correctly classify all samples in the ROC.
        for k in range(len(y), 0, -1):
            mask = np.all(oracle[:k,:], axis=0)
            if np.any(mask):
                [selected_idx] = np.where(mask)
                break

        # if no classifier was selected, select the single best
        # classifier and all classifiers that corretly classify
        # the same number of samples as this classifier.
        if selected_idx is None:
            scores = np.sum(oracle, axis=0)
            [selected_idx] = np.where(scores == np.max(scores))

        selected = [ensemble._estimators[i] for i in selected_idx]
        return Ensemble(estimators=selected), None

class KNORA_UNION(DCS):
    """K-nearest-oracles Union.

    The KNORA union reduces the neighborhood until finds an
    ensemble of classifiers that correctly classify all neighbors.

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

    weighted : book, optional (default = False)
        If True, the samples in the region of competence are weighted using
        the inverse of their distances to the test sample.
        If False, all samples in the reigon of competence have the same weight.

    See also
    --------
    skensemble.selection.dynamic.knora.KNORA_ELIMINATE: KNORA Eliminate.

    References
    ----------
    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira.
    "Dynamic selection of classifiers—A comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.
    """
    def __init__(self, roc_selector=None, roc_size=7, weighted=False):
        super(KNORA_UNION, self).__init__(roc_selector, roc_size)
        self.weighted = weighted

    def _select(self, ensemble, x):
        X, y, dists = self.get_roc(x, return_distance=True)

        # oracle shape (n_samples, n_estimators)
        oracle = ensemble.oracle(X, y)
        [selected_idx] = np.where(np.any(oracle, axis=0))

        if len(selected_idx) > 0:
            if self.weighted:
                weights = 1. / (dists + 10e-24)
            else:
                weights = np.ones_line(y)

            votes = oracle[:,selected_idx].astype(float) * weights[:, np.newaxis]
            weighted_votes = np.sum(votes, axis=0)
            selected = Ensemble([ensemble._estimators[i] for i in selected_idx])
        else:
            selected = Ensemble(estimators=ensemble._estimators)
            weighted_votes = None

        return selected, weighted_votes

