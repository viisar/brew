import numpy as np

from sklearn.metrics import accuracy_score

from brew.combination.combiner import Combiner
from brew.metrics.evaluation import auc_score


def transform2votes(output, n_classes):

    n_samples = output.shape[0]

    votes = np.zeros((n_samples, n_classes), dtype=int)
    # uses the predicted label as index for the vote matrix
    # for i in range(n_samples):
    #    idx = int(output[i])
    #    votes[i, idx] = 1
    votes[np.arange(n_samples), output.astype(int)] = 1
    # assert np.equal(votes2.astype(int), votes.astype(int)).all()

    return votes.astype(int)


class Transformer(object):

    def __init__(self):
        pass

    def apply(self, X):
        pass


class FeatureSubsamplingTransformer(Transformer):

    def __init__(self, features=None):
        self.features = features

    def apply(self, X):
        # if is only one sample (1D)
        if X.ndim == 1:
            return X[self.features]
        # if X has more than one sample (2D)
        else:
            return X[:, self.features]


class BrewClassifier(object):

    def __init__(self, classifier=None, transformer=None):
        self.transformer = transformer
        self.classifier = classifier
        self.classes_ = []

    def fit(self, X, y):
        X = self.transformer.apply(X)
        self.classifier.fit(X, y)
        self.classes_ = self.classifier.classes_

    def predict(self, X):
        X = self.transformer.apply(X)
        y = self.classifier.predict(X)
        return y

    def predict_proba(self, X):
        X = self.transformer.apply(X)
        y = self.classifier.predict_proba(X)
        return y


class Ensemble(object):
    """Class that represents a collection of classifiers.

    The Ensemble class serves as a wrapper for a list of classifiers,
    besides providing a simple way to calculate the output of all the
    classifiers in the ensemble.

    Attributes
    ----------
    `classifiers` : list
        Stores all classifiers in the ensemble.

    `yval` : array-like, shape = [indeterminated]
        Labels of the validation set.

    `knn`  : sklearn KNeighborsClassifier,
        Classifier used to find neighborhood.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> from brew.base import Ensemble
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0],
                      [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>>
    >>> dt1 = DecisionTreeClassifier()
    >>> dt2 = DecisionTreeClassifier()
    >>>
    >>> dt1.fit(X, y)
    >>> dt2.fit(X, y)
    >>>
    >>> ens = Ensemble(classifiers=[dt1, dt2])

    """

    def __init__(self, classifiers=None):

        if classifiers is None:
            self.classifiers = []
        else:
            self.classifiers = classifiers

    def add(self, classifier):
        self.classifiers.append(classifier)

    def add_classifiers(self, classifiers):
        self.classifiers = self.classifiers + classifiers

    def add_ensemble(self, ensemble):
        self.classifiers = self.add_classifiers(ensemble.classifiers)

    def get_classes(self):
        classes = set()
        for c in self.classifiers:
            classes = classes.union(set(c.classes_))

        self.classes_ = list(classes)
        return self.classes_

    def output(self, X, mode='votes'):
        """Returns the output of all classifiers packed in a numpy array.

        This method calculates the output of each classifier, and stores
        them in a array-like shape. The specific shape and the meaning of
        each element is defined by argument `mode`.

        (1) 'labels': each classifier will return a single label
        prediction for each sample in X, therefore the ensemble
        output will be a 2d-array of shape (n_samples, n_classifiers),
        with elements being the class labels.

        (2) 'probs': each classifier will return the posterior
        probabilities of each class (i.e. instead of returning
        a single choice it will return the probabilities of each
        class label being the right one). The ensemble output
        will be a 3d-array with shape (n_samples, n_classes,
        n_classifiers), with each element being the probability
        of a specific class label being right on a given sample
        according to one the classifiers. This mode can be used
        with any combination rule.

        (3) 'votes': each classifier will return votes for each
        class label i.e. a binary representation, where the chosen
        class label will have one vote and the other labels will
        have zero votes. The ensemble output will be a binary
        3d-array with shape (n_samples, n_classes, n_classifiers),
        with the elements being the votes. This mode is mainly
        used in combining the classifiers output by using majority
        vote rule.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
                The test input samples.

        mode: string, optional(default='labels')
                The type of output given by each classifier.
                'labels' | 'probs' | 'votes'
        """

        if mode == 'labels':
            out = np.zeros((X.shape[0], len(self.classifiers)))
            for i, clf in enumerate(self.classifiers):
                out[:, i] = clf.predict(X)

        else:
            # assumes that all classifiers were
            # trained with the same number of classes
            classes__ = self.get_classes()
            n_classes = len(classes__)
            out = np.zeros((X.shape[0], n_classes, len(self.classifiers)))

            for i, c in enumerate(self.classifiers):
                if mode == 'probs':
                    probas = np.zeros((X.shape[0], n_classes))
                    probas[:, list(c.classes_)] = c.predict_proba(X)
                    out[:, :, i] = probas

                elif mode == 'votes':
                    tmp = c.predict(X)  # (n_samples,)
                    # (n_samples, n_classes)
                    votes = transform2votes(tmp, n_classes)
                    out[:, :, i] = votes

        return out

    def output_simple(self, X):
        out = np.zeros((X.shape[0], len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            out[:, i] = clf.predict(X)

        return out

    def in_agreement(self, x):
        prev = None
        for clf in self.classifiers:
            [tmp] = clf.predict(x)
            if tmp != prev:
                return False
            prev = tmp

        return True

    def __len__(self):
        return len(self.classifiers)

    def fit(self, X, y):
        '''
        warning: this fit overrides previous generated base classifiers!
        '''
        for clf in self.classifiers:
            clf.fit(X, y)

        return self


class EnsembleClassifier(object):

    def __init__(self, ensemble=None, selector=None, combiner=None):
        self.ensemble = ensemble
        self.selector = selector

        if combiner is None:
            self.combiner = Combiner(rule='majority_vote')
        elif isinstance(combiner, str):
            self.combiner = Combiner(rule=combiner)
        elif isinstance(combiner, Combiner):
            self.combiner = combiner
        else:
            raise ValueError('Invalid parameter combiner')

    def fit(self, X, y):
        self.ensemble.fit(X, y)

    def predict(self, X):

        # TODO: warn the user if mode of ensemble
        # output excludes the chosen combiner?

        if self.selector is None:
            out = self.ensemble.output(X)
            y = self.combiner.combine(out)

        else:
            y = []

            for i in range(X.shape[0]):
                ensemble, weights = self.selector.select(
                    self.ensemble, X[i, :][np.newaxis, :])

                if weights is not None:  # use the ensemble with weights
                    if self.combiner.combination_rule == 'majority_vote':
                        out = ensemble.output(X[i, :][np.newaxis, :])
                    else:
                        out = ensemble.output(X[i, :][np.newaxis, :], mode='probs')

                    # apply weights
                    for i in range(out.shape[2]):
                        out[:, :, i] = out[:, :, i] * weights[i]

                    [tmp] = self.combiner.combine(out)
                    y.append(tmp)

                else:  # use the ensemble, but ignore the weights
                    if self.combiner.combination_rule == 'majority_vote':
                        out = ensemble.output(X[i, :][np.newaxis, :])
                    else:
                        out = ensemble.output(X[i, :][np.newaxis, :], mode='probs')
                    [tmp] = self.combiner.combine(out)
                    y.append(tmp)

        return np.asarray(y)

    def predict_proba(self, X):

        # TODO: warn the user if mode of ensemble
        # output excludes the chosen combiner?

        if self.selector is None:
            out = self.ensemble.output(X, mode='probs')
            return np.mean(out, axis=2)

        else:
            out_full = []

            for i in range(X.shape[0]):
                ensemble, weights = self.selector.select(
                    self.ensemble, X[i, :][np.newaxis, :])

                if weights is not None:  # use the ensemble with weights
                    out = ensemble.output(X[i, :][np.newaxis, :])

                    # apply weights
                    for i in range(out.shape[2]):
                        out[:, :, i] = out[:, :, i] * weights[i]

                    # [tmp] = self.combiner.combine(out)
                    out_full.extend(list(np.mean(out, axis=2)))

                else:  # use the ensemble, but ignore the weights
                    out = ensemble.output(X[i, :][np.newaxis, :])
                    out_full.extend(list(np.mean(out, axis=2)))

        # return np.asarray(y)
        return np.array(out_full)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


def oracle(ensemble, X, y_true, metric=auc_score):
    out = ensemble.output(X, mode='labels')
    oracle = np.equal(out, y_true[:, np.newaxis])
    mask = np.any(oracle, axis=1)
    y_pred = out[:, 0]
    y_pred[mask] = y_true[mask]
    return metric(y_pred, y_true)


def single_best(ensemble, X, y_true, metric=auc_score):
    out = ensemble.output(X, mode='labels')
    scores = np.zeros(len(ensemble), dtype=float)
    for i in range(scores.shape[0]):
        scores[i] = metric(out[:, i], y_true)
    return np.max(scores)
