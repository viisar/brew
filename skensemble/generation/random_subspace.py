import numpy as np

import sklearn

from brew.base import Ensemble
from brew.base import FeatureSubsamplingTransformer
from brew.base import BrewClassifier
from brew.combination.combiner import Combiner
from .base import PoolGenerator


class RandomSubspace(PoolGenerator):

    def __init__(self,
                 base_classifier=None,
                 n_classifiers=100,
                 combination_rule='majority_vote',
                 max_features=0.5):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.combiner = Combiner(rule=combination_rule)
        self.classifiers = None
        self.ensemble = None
        self.max_features = max_features

    def fit(self, X, y):
        self.ensemble = Ensemble()

        for i in range(self.n_classifiers):
            chosen_features = np.random.choice(X.shape[1], int(
                np.ceil(X.shape[1] * self.max_features)), replace=False)
            transformer = FeatureSubsamplingTransformer(
                features=chosen_features)

            classifier = BrewClassifier(classifier=sklearn.base.clone(
                self.base_classifier), transformer=transformer)
            classifier.fit(X, y)

            self.ensemble.add(classifier)

        return

    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)
