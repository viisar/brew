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

        if knn is None:
            self.knn = KNeighborsClassifier(n_neighbors=K, algorithm='brute')
        else:
            self.knn = knn

        self.knn.fit(Xval, yval)
        self.weighted = weighted

    def get_neighbors(self, x, return_distance=False):
        # obtain the K nearest neighbors of test sample in the validation set
        if not return_distance:
            [idx] = self.knn.kneighbors(x,
                                        return_distance=return_distance)
        else:
            rd = return_distance
            [dists], [idx] = self.knn.kneighbors(x, return_distance=rd)
        X_nn = self.Xval[idx]  # k neighbors
        y_nn = self.yval[idx]  # k neighbors target

        if return_distance:
            return X_nn, y_nn, dists
        else:
            return X_nn, y_nn
