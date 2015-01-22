import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier

from brew.base import EnsembleClassifier

class DCS(EnsembleClassifier):

    def select(self, x, ensemble):
        pass

    def __init__(self, Xval, yval, K=5):
        self.Xval = Xval
        self.yval = yval
        self.K = K
        self.knn = KNeighborsClassifier(n_neighbors=K, algorithm='brute')
        self.knn.fit(Xval, yval)

