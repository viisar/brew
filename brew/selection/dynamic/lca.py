import numpy as np

from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier

from brew.base import Ensemble
from .base import DCS

class LCA(DCS):

    def predict(self, X):
        X_tst = np.array(X)
        y_pred = []

        for i in range(X_tst.shape[0]):
            if self.in_agreement(X_tst[i]):
                y_pred += [self.classifiers[0].predict(X_tst[i])]
            else:
                [idx] = self.knn.kneighbors(X_tst[i], return_distance=False)
                mx_id, mx_vl = -1, -1
                for e, clf in enumerate(self.classifiers):
                    right, count = 0, 0
                    for xv, yv in zip(self.val_X[idx], self.val_y[idx]):
                        pred = clf.predict(xv)
                        if pred == clf.predict(X_tst[i]):
                            count = count + 1
                            if pred == yv:
                                right = right + 1
                    if right > 0 and count > 0 and float(right)/count > mx_vl:
                        mx_id, mx_vl = e, float(right)/count
                        
                y_pred += [self.classifiers[mx_id].predict(X_tst[i])]

        return np.array(y_pred)


def LCA2(DCS):
    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]])

        classifiers = ensemble.classifiers
        [idx] = self.knn.kneighbors(x, return_distance=False)
        X, y = self.Xval[idx], self.yval[idx]

        









