import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier

from abc import abstractmethod

class DCS(object):

    @abstractmethod
    def select(self, ensemble, x):
        pass

    def __init__(self, Xval, yval, K=5, weighted=False, knn=None):
        self.Xval = Xval
        self.yval = yval
        self.K = K

        if knn == None:
            self.knn = KNeighborsClassifier(n_neighbors=K, algorithm='brute')
        else:
            self.knn = knn

        self.knn.fit(Xval, yval)
        self.weighted = weighted

