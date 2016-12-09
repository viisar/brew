import numpy as np

from brew.base import Ensemble
from .base import DCS


class LCA2(DCS):
    """Local Class Accuracy.

    The Local Class Accuracy selects the best classifier for
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
    >>> from brew.selection.dynamic.lca import LCA
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
    >>> lca = LCA(X, y, K=3)
    >>>
    >>> clf = EnsembleClassifier(bag.ensemble, selector=lca)
    >>> clf.predict([-1.1,-0.5])
    [1]

    See also
    --------
    brew.selection.dynamic.ola.OLA: Overall Local Accuracy.

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

    def __init__(self, Xval, yval, K=5, weighted=False, knn=None):
        '''
        Parameters
        ----------
        Xval : Numpy 2d-array with rows representing each sample.
        yval : Numpy 1d-array representing the target classes of
            the samples in Xval.
        K : int (default=5), the size of the neighborhood used to select the
            classifier.
        weighted : bool (default=False), if the selected classifiers are
            weighted;
        knn : sklearn KNeighborsClassifier (default=None), a classifier
            to find the neighborhood of each sample.
        '''
        super(LCA2, self).__init__(Xval, yval, K, weighted, knn)

    def select(self, ensemble, x):

        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]]), None

        # obtain the K nearest neighbors in the validation set
        [idx] = self.knn.kneighbors(x, return_distance=False)
        neighbors_X = self.Xval[idx]  # k neighbors
        neighbors_y = self.yval[idx]  # k neighbors target

        # pool_output (sample, classifier_output)
        pool_output = np.zeros((neighbors_X.shape[0], len(ensemble)))
        for i, clf in enumerate(ensemble.classifiers):
            pool_output[:, i] = clf.predict(neighbors_X)

        x_outputs = [ensemble.classifiers[j].predict(
            x) for j in range(len(ensemble))]
        x_outputs = np.asarray(x_outputs).flatten()

        d = {}
        scores = np.zeros(len(ensemble))
        for j in range(pool_output.shape[1]):
            # get correctly classified samples
            mask_classified_correctly = pool_output[:, j] == neighbors_y
            # get classified samples with the same class as 'x'
            mask_classified_same_class = (pool_output[:, j] == x_outputs[j])
            # get correctly classified samples with the same class as 'x'
            mask = mask_classified_correctly * mask_classified_same_class
            # calculate score
            scores[j] = float(sum(mask)) / (sum(mask_classified_same_class) + 10e-24)
            d[str(scores[j])] = d[str(scores[j])] + [j] if str(scores[j]) in d else [j]


        best_scores = sorted([float(k) for k in list(d.keys())], reverse=True)

        options = None
        for j, score in enumerate(best_scores):
            pred = [x_outputs[i] for i in d[str(score)]]
            pred = np.asarray(pred).flatten()

            bincount = np.bincount(pred.astype(int))
            if options is not None:
                for i in range(len(bincount)):
                    bincount[i] = bincount[i] if i in options else 0

            imx = np.argmax(bincount)
            votes = np.argwhere(bincount == bincount[imx]).flatten()
            count = len(votes)
            if count == 1:
                ens = Ensemble([ensemble.classifiers[np.argmax(pred == imx)]])
                return ens, None
            elif options is None:
                options = votes

        return Ensemble([ensemble.classifiers[np.argmax(scores)]]), None

class LCA(DCS):

    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]]), None

        # obtain the K nearest neighbors in the validation set
        [idx] = self.knn.kneighbors(x, n_neighbors=self.K, return_distance=False)
        neighbors_X = self.Xval[idx] # k neighbors
        neighbors_y = self.yval[idx] # k neighbors target

        # pool_output (sample, classifier_output)
        pool_output = np.zeros((neighbors_X.shape[0], len(ensemble)))
        for i, clf in enumerate(ensemble.classifiers):
            pool_output[:, i] = clf.predict(neighbors_X)

        x_outputs = [ensemble.classifiers[j].predict(
            x) for j in range(len(ensemble))]
        x_outputs = np.asarray(x_outputs).flatten()

        scores = np.zeros(len(ensemble))
        for j in range(pool_output.shape[1]):
            # get correctly classified samples
            mask_classified_correctly = pool_output[:, j] == neighbors_y
            # get classified samples with the same class as 'x'
            mask_classified_same_class = (pool_output[:, j] == x_outputs[j])
            # get correctly classified samples with the same class as 'x'
            mask = mask_classified_correctly * mask_classified_same_class
            # calculate score
            scores[j] = float(sum(mask)) / (sum(mask_classified_same_class) + 10e-24)

        return Ensemble([ensemble.classifiers[np.argmax(scores)]]), None

