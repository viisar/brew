import numpy as np

from brew.base import Ensemble
from .base import DCS


class LCA(DCS):
    """Local Class Accuracy.

    The Overall Local Accuracy selects the best classifier for
    a sample using it's K nearest neighbors.

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

    tie_mode : str, how the competence ties are handled, optional (default = 'ignore')
        - If 'ignore', randomly selects one of the  most competent estimators.
        - If 'break', majority vote is used to break ties.

        NOTE: 'ignore' is faster.

    See also
    --------
    skensemble.selection.dynamic.lca.LCA: Local Class Accuracy.

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
    def __init__(self, roc_selector=None, roc_size=7, tie_mode='ignore'):
        super(LCA, self).__init__(roc_selector, roc_size)

        if tie_mode not in VALID_TIE_MODES:
            raise NotImplementedError

        self.tie_mode = tie_mode

    def _select(self, ensemble, x):
        X, y = self.get_roc(x, return_distance=False)

        output = ensemble.output(X)
        labels = output2labels(output, classes=ensemble.classes_)
        test_labels = output2labels(ensemble.output(x.reshape(1,-1)),
                ensemble.classes_).ravel()
        # labels shape (n_samples, n_estimators)

        scores = np.zeros(len(ensemble))
        for i in range(len(ensemble)):
            # get correctly classified samples
            mask_classified_correctly = (labels[:, i] == y)
            # get classified samples with the same class as 'x'
            mask_classified_same_class = (labels[:, i] == test_labels[i])
            # get correctly classified samples with the same class as 'x'
            mask = np.logical_and(mask_classified_correctly, mask_classified_same_class)
            # calculate score
            scores[i] = float(np.sum(mask)) / (np.sum(mask_classified_same_class) + 10e-24)

        assert len(scores) == len(ensemble)

        if self.tie_mode == 'ignore':
            best_estimator = ensemble._estimators[np.argmax(scores)]
            return Ensemble([best_estimator]), None

        elif self.tie_mode == 'break':
            options = None

            test_labels = output2labels(ensemble.output(x.reshape(1,-1)),
                    ensemble.classes_)

            [high_score_idx] = np.where(scores == np.max(scores))

            for i, score_level in enumerate(sorted(set(score), reverse=True)):
                [idx] = np.where(scores == score_level)
                consensus = get_consensus(test_labels[idx], options)
                if consensus is not None:
                    [c_idx] = np.where(test_labels[high_score_idx] == consensus)
                    best_estimator_idx = high_score_idx[c_idx[0]]
                    best_estimator = ensemble._estimators[best_estimator_idx]
                    return Ensemble([best_estimator])
                elif options is None:
                    options = set(test_labels[idx])
                else:
                    ties = get_ties(test_labels[idx], options=options)
                    options = options.union(set(ties))

            # if has not selected any use random selection criterion
            best_estimator = ensemble._estimators[np.argmax(scores)]
            return Ensemble([best_estimator]), None


