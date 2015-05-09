from brew.base import Ensemble
from brew.base import FeatureSubsamplingTransformer
from brew.base import BrewClassifier
from brew.combination.combiner import Combiner

from sklearn.tree import DecisionTreeClassifier
import sklearn

from brew.combination.combiner import Combiner
from brew.combination.rules import majority_vote_rule
from .base import PoolGenerator

import numpy as np
from sklearn.ensemble import BaggingClassifier


class RandomSubspace(PoolGenerator):

    def __init__(self, base_classifier=None, n_classifiers=100, combination_rule=majority_vote_rule, max_features=0.5):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.combination_rule = combination_rule
        self.sk_random_subspace = BaggingClassifier(base_estimator=base_classifier,
                n_estimators=n_classifiers, max_samples=1.0, max_features=max_features)
        self.classifiers = None
        self.ensemble = None
        

    def fit(self, X, y):
        self.X = X
        self.y = y

        self.sk_random_subspace.fit(X, y)
        self.classifiers = self.sk_random_subspace.estimators_
        self.ensemble = Ensemble(classifiers=self.classifiers)
        self.classes_ = set(y)

        return self


    def predict(self, X):
        #TODO usar combinator
        y = []
        for i in range(X.shape[0]):
            d = {}
            mx = None
            for idx, classifier in enumerate(self.classifiers):
                mask = np.zeros(X.shape[1], bool)
                mask[self.sk_random_subspace.estimators_features_[idx]] = True
                [out] = classifier.predict(X[i][mask])
                d[out] = d[out] + 1 if out in d else 1
                if mx == None or d[mx] < d[out]:
                    mx = out
            y = y + [mx]
        y = map(lambda e: self.sk_random_subspace.classes_[e], y)

        return np.asarray(y)
            
                
class RandomSubspaceNew(PoolGenerator):

    def __init__(self, base_classifier=None, n_classifiers=100, combination_rule='majority_vote', max_features=0.5):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.combiner = Combiner(rule=combination_rule)
        self.classifiers = None
        self.ensemble = None
        self.max_features = max_features
       
    def fit(self, X, y):
        self.ensemble = Ensemble()

        for i in range(self.n_classifiers):
            chosen_features = np.random.choice(X.shape[1], int(X.shape[1]*self.max_features), replace=False)
            transformer = FeatureSubsamplingTransformer(features=chosen_features)

            classifier = BrewClassifier(classifier=sklearn.base.clone(self.base_classifier), transformer=transformer)
            classifier.fit(X, y)
            
            self.ensemble.add(classifier)

        return

    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)


if __name__ == '__main__':
    
    X = np.random.random((10,4))
    y = np.random.randint(0,2,10)

    pool = RandomSubspaceNew(base_classifier=DecisionTreeClassifier(), n_classifiers=40)
    pool.fit(X, y)

    print pool.predict(X)
