from brew.base import Ensemble
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

        class RandSubClf(object):
            def __init__(self, mask, clf):
                self.mask = mask
                self.clf = clf
                self.classes_ = clf.classes_

            def predict(self, X):
                return self.clf.predict(X[:,self.mask])
                
        classifiers = []
        for idx, clf in enumerate(self.classifiers):
            mask = np.zeros(X.shape[1], bool)
            mask[self.sk_random_subspace.estimators_features_[idx]] = True
            classifiers = classifiers + [RandSubClf(mask, clf)]

        self.ensemble = Ensemble(classifiers=classifiers)
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
            
                
                
        
