from brew.base import Ensemble
from brew.combination.rules import majority_vote_rule
from .base import PoolGenerator

import numpy as np
from sklearn.ensemble import BaggingClassifier

class Bagging(PoolGenerator):

    def __init__(self, base_classifier=None, n_classifiers=100, combination_rule=majority_vote_rule):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.combination_rule = combination_rule
        self.sk_bagging = BaggingClassifier(base_estimator=base_classifier,
                n_estimators=n_classifiers, max_samples=1.0, max_features=1.0)
        self.classifiers = None
        self.ensemble = None
        

    def fit(self, X, y):
        self.sk_bagging.fit(X, y)
        self.classifiers = self.sk_bagging.estimators_
        self.ensemble = Ensemble(classifiers=self.classifiers)
        self.classes_ = set(y)


    def predict(self, X):
        #TODO usar combinator
        y = []
        for i in range(X.shape[0]):
            d = {}
            mx = None
            for classifier in self.classifiers:
                [out] = classifier.predict(X[i])
                d[out] = d[out] + 1 if out in d else 1
                if mx == None or d[mx] < d[out]:
                    mx = out
            y = y + [mx]
        y = map(lambda e: self.sk_bagging.classes_[e], y)

        return np.asarray(y)
            
                
                
        
