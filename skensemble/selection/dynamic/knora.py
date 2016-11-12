# -*- coding: utf-8 -*-

import numpy as np

from .base import DCS

from brew.base import Ensemble


# do not use this class directly, call it's subclasses instead (e.g. KNORA_E)
class KNORA(DCS):

    def _get_best_classifiers(self, ensemble, neighbors_X, neighbors_y, x):
        ensemble_out = ensemble.output(neighbors_X, mode='labels')
        ensemble_mask = ensemble_out == neighbors_y[:, np.newaxis]

        correct = np.sum(ensemble_mask, axis=0)
        idx = np.argmax(correct)  # best classifier idx

        all_idx = correct == correct[idx]

        pool = [ensemble.classifiers[i] for i in all_idx]

        return pool


class KNORA_ELIMINATE(KNORA):
    """K-nearest-oracles Eliminate.

    The KNORA Eliminate reduces the neighborhood until finds an
    ensemble of classifiers that correctly classify all neighbors.

    Attributes
    ----------
    `Xval` : array-like, shape = [indeterminated, n_features]
        Validation set.

    `yval` : array-like, shape = [indeterminated]
        Labels of the validation set.

    `knn`  : sklearn KNeighborsClassifier,
        Classifier used to find neighborhood.

    `weighted` : bool, (makes no difference in knora_eliminate)
        Bool that defines if the classifiers uses weights or not


    Examples
    --------
    >>> from brew.selection.dynamic.knora import KNORA_ELIMINATE
    >>> from brew.generation.bagging import Bagging
    >>> from brew.base import EnsembleClassifier
    >>>
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> import numpy as np
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0],
                      [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>>
    >>> dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    >>> bag = Bagging(base_classifier=dt, n_classifiers=10)
    >>> bag.fit(X, y)
    >>>
    >>> ke = KNORA_ELIMINATE(X, y, K=5)
    >>>
    >>> clf = EnsembleClassifier(bag.ensemble, selector=ke)
    >>> clf.predict([-1.1,-0.5])
    [1]

    See also
    --------
    brew.selection.dynamic.knora.KNORA_UNION: KNORA Union.
    brew.selection.dynamic.lca.LCA: Local Class Accuracy.
    brew.selection.dynamic.ola.OLA: Overall Local Accuracy.

    References
    ----------
    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira.
    "Dynamic selection of classifiers—A comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    Hung-Ren Ko, A., Robert Sabourin, and A. de Souza Britto.
    "K-nearest oracle for dynamic ensemble selection."
    Document Analysis and Recognition, 2007. ICDAR 2007.
    Ninth International Conference on. Vol. 1. IEEE, 2007

    """

    def __init__(self, Xval, yval, K=5, weighted=False, knn=None, v2007=False):
        self.v2007 = v2007
        super(KNORA_ELIMINATE, self).__init__(
            Xval, yval, K=K, weighted=weighted, knn=knn)

    def select(self, ensemble, x):
        ensemble_mask = None

        neighbors_X, neighbors_y = self.get_neighbors(x)
        pool_output = ensemble.output(neighbors_X, mode='labels')

        # gradually decrease neighborhood size if no
        # classifier predicts ALL the neighbors correctly
        for i in range(self.K, 0, -1):
            pool_mask = _get_pool_mask(
                pool_output[:i], neighbors_y[:i], np.all)

            # if at least one classifier gets all neighbors right
            if pool_mask is not None:
                ensemble_mask = pool_mask
                break

        # if NO classifiers get the nearest neighbor correctly
        if ensemble_mask is None:

            if self.v2007:
                # Increase neighborhood until one classifier
                # gets at least ONE (i.e. ANY) neighbors correctly.
                # Starts with 2 because mask_all with k=1 is
                # the same as mask_any with k=1
                for i in range(2, self.K + 1):
                    pool_mask = _get_pool_mask(
                        pool_output[:i], neighbors_y[:i], np.any)

                    if pool_mask is not None:
                        ensemble_mask = pool_mask
                        break

        [selected_idx] = np.where(ensemble_mask)

        if selected_idx.size > 0:
            pool = Ensemble(
                classifiers=[ensemble.classifiers[i] for i in selected_idx])

        else:  # use all classifiers
            # pool = ensemble
            classifiers = self._get_best_classifiers(
                ensemble, neighbors_X, neighbors_y, x)
            pool = Ensemble(classifiers=classifiers)

        # KNORA-ELIMINATE-W that supposedly uses weights, does not make
        # any sense, so even if self.weighted is True, always return
        # None for the weights

        return pool, None


class KNORA_UNION(KNORA):
    """K-nearest-oracles Union.

    The KNORA union reduces the neighborhood until finds an
    ensemble of classifiers that correctly classify all neighbors.

    Attributes
    ----------
    `Xval` : array-like, shape = [indeterminated, n_features]
        Validation set.

    `yval` : array-like, shape = [indeterminated]
        Labels of the validation set.

    `knn`  : sklearn KNeighborsClassifier,
        Classifier used to find neighborhood.

    `weighted` : bool, (makes no difference in knora_eliminate)
        Bool that defines if the classifiers uses weights or not


    Examples
    --------
    >>> from brew.selection.dynamic.knora import KNORA_UNION
    >>> from brew.generation.bagging import Bagging
    >>> from brew.base import EnsembleClassifier
    >>>
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> import numpy as np
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0],
                      [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>>
    >>> dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    >>> bag = Bagging(base_classifier=dt, n_classifiers=10)
    >>> bag.fit(X, y)
    >>>
    >>> ku = KNORA_UNION(X, y, K=5)
    >>>
    >>> clf = EnsembleClassifier(bag.ensemble, selector=ku)
    >>> clf.predict([-1.1,-0.5])
    [1]

    See also
    --------
    brew.selection.dynamic.knora.KNORA_ELIMINATE: Knora Eliminate.
    brew.selection.dynamic.lca.LCA: Local Class Accuracy.
    brew.selection.dynamic.ola.OLA: Overall Local Accuracy.

    References
    ----------
    Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
    "From dynamic classifier selection to dynamic ensemble selection."
    Pattern Recognition 41.5 (2008): 1718-1731.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira.
    "Dynamic selection of classifiers—A comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    Hung-Ren Ko, A., Robert Sabourin, and A. de Souza Britto.
    "K-nearest oracle for dynamic ensemble selection."
    Document Analysis and Recognition, 2007. ICDAR 2007.
    Ninth International Conference on. Vol. 1. IEEE, 2007.
    """

    def select(self, ensemble, x):
        neighbors_X, neighbors_y = self.get_neighbors(x)
        pool_output = ensemble.output(neighbors_X, mode='labels')

        output_mask = (pool_output == neighbors_y[:, np.newaxis])

        [selected_idx] = np.where(np.any(output_mask, axis=0))

        if selected_idx.size > 0:
            if self.weighted:
                weights = 1.0 / \
                    (np.sqrt(np.sum((x - neighbors_X)**2, axis=1)) + 10e-8)
                weighted_votes = np.dot(weights, output_mask[:, selected_idx])
            else:
                weighted_votes = np.sum(output_mask[:, selected_idx], axis=0)

            pool = Ensemble(
                classifiers=[ensemble.classifiers[i] for i in selected_idx])
        # if no classifiers are selected,
        # use all classifiers with no weights
        else:
            pool = ensemble
            weighted_votes = None

        return pool, weighted_votes


def _get_pool_mask(pool_output, neighbors_target, func):
    pool_mask = func(pool_output == neighbors_target[:, np.newaxis], axis=0)

    if np.sum(pool_mask) > 0:
        return pool_mask

    return None
