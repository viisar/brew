import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier

from .base import DCS

class OLA(DCS):

    def predict(self, X):
        X_tst = np.array(X)
        y_pred = []
        for i in range(X_tst.shape[0]):
            if self.in_agreement(X_tst[i]):
                y_pred += [self.classifiers[0].predict(X_tst[i])]
            else:
                [idx] = self.knn.kneighbors(X_tst[i], return_distance=False)
                scores = [clf.score(self.val_X[idx], self.val_y[idx]) for clf in self.classifiers]
                clf = self.classifiers[np.argmax(scores)]
                y_pred += [clf.predict(X_tst[i])]

        return np.array(y_pred)

class OLA2(DCS):

    def predict(self, X):
        X_tst = np.array(X)

        X_idxs = self.knn.kneighbors(X_tst, return_distance=False)
        c_idxs = map(lambda e: np.argmax([clf.score(self.val_X[e], self.val_y[e]) for clf in self.classifiers]), X_idxs)
        y_pred = [self.classifiers[idx].predict(X_tst[i]) for (idx, i) in zip(c_idxs, range(X_tst.shape[0]))]

        return np.asarray(y_pred)

