import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier


class DCS(object):

    def __init__(self, classifiers, val_X, val_y, K=5):
        self.classifiers = classifiers
        self.val_X = val_X
        self.val_y = val_y
        self.K = K
        self.knn = KNeighborsClassifier(n_neighbors=K, algorithm='brute')
        self.knn.fit(val_X, val_y)

    def predict(self, X):
        pass

    def in_agreement(self, X_i):
        prev = None
        for clf in self.classifiers:
            tmp = clf.predict(X_i)
            if tmp != prev:
                return False
            prev = tmp

        return True
