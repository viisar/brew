import numpy as np

import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA

from brew.base import Ensemble
from brew.combination.rules import majority_vote_rule
from brew.generation import RandomSubspace

from .base import PoolGenerator

class RandomNewspace(PoolGenerator):

    def __init__(self, K=10, bootstrap_samples=0.75, bootstrap_features=0.75, base_classifier=None, n_classifiers=100,
            combination_rule=majority_vote_rule, max_samples=1.0, max_features=0.5):

        self.K = K
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_features = bootstrap_features        

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.combination_rule = combination_rule
        self.random_subspace = RandomSubspace(base_classifier=base_classifier, n_classifiers=n_classifiers, 
                combination_rule=combination_rule, max_features=max_features)
        self.random_subspace.sk_random_subspace.max_samples = max_samples

        self.pca_transformers = [PCA(n_components=1) for i in range(self.K)]
        self.lda_transformers = [LDA() for i in range(self.K)]
        self.mask_transformers = None
        self.classifiers = None
        self.ensemble = None
        

    def fit(self, X, y):
        new_X = np.array(X)
        new_y = np.array(y)

        self.mask_transformers = []

        for i in range(self.K):
            tmp, sX, tmp, sy = sklearn.cross_validation.train_test_split(X, y, test_size=self.bootstrap_samples)

            mask = np.ceil(self.bootstrap_features * X.shape[1]) * [True]
            mask = mask + (X.shape[1] - len(mask)) * [False]
            np.random.shuffle(mask)
            mask = np.array(mask, dtype=bool)

            self.mask_transformers += [mask]
            #print (i, self.K), (len(X), len(sX)), (len(X[0]), len(mask))
            sX = sX[:,mask]
            
            self.pca_transformers[i].fit(sX)
            self.lda_transformers[i].fit(sX, sy)

            new_features = self.pca_transformers[i].transform(X[:,mask])
            new_X = np.concatenate((new_X, new_features), axis=1)
            new_features = self.lda_transformers[i].transform(X[:,mask])
            new_X = np.concatenate((new_X, new_features), axis=1)


        self.random_subspace.fit(new_X, y)
        self.classifiers = self.random_subspace.classifiers
        self.ensemble = self.random_subspace.ensemble
        self.classes_ = self.random_subspace.classes_

        return self

    def predict(self, X):
        X_tst = np.array(X)
        for i in range(self.K):
            mask = self.mask_transformers[i]
            new_features = self.pca_transformers[i].transform(X[:,mask])
            X_tst = np.concatenate((X_tst, new_features), axis=1)
            new_features = self.lda_transformers[i].transform(X[:,mask])
            X_tst = np.concatenate((X_tst, new_features), axis=1)
            
        return self.random_subspace.predict(X_tst)
       
