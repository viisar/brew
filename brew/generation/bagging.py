import numpy as np
from sklearn.ensemble import BaggingClassifier

from brew.base import Ensemble
from brew.combination.combiner import Combiner
import sklearn

from .base import PoolGenerator


class Bagging(PoolGenerator):

    def __init__(self,
                 base_classifier=None,
                 n_classifiers=100,
                 combination_rule='majority_vote'):

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.ensemble = None
        self.combiner = Combiner(rule=combination_rule)

    def fit(self, X, y):
        self.ensemble = Ensemble()

        for _ in range(self.n_classifiers):
            # bootstrap
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            data, target = X[idx, :], y[idx]

            classifier = sklearn.base.clone(self.base_classifier)
            classifier.fit(data, target)

            self.ensemble.add(classifier)

        return

    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)


class BaggingSK(PoolGenerator):
    """"
    This class should not be used, use brew.generation.bagging.Bagging instead.
    """

    def __init__(self,
                 base_classifier=None,
                 n_classifiers=100,
                 combination_rule='majority_vote'):

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers

        # using the sklearn implementation of bagging for now
        self.sk_bagging = BaggingClassifier(base_estimator=base_classifier,
                                            n_estimators=n_classifiers,
                                            max_samples=1.0,
                                            max_features=1.0)

        self.ensemble = Ensemble()
        self.combiner = Combiner(rule=combination_rule)

    def fit(self, X, y):
        self.sk_bagging.fit(X, y)
        self.ensemble.add_classifiers(self.sk_bagging.estimators_)
        # self.classes_ = set(y)

    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)
