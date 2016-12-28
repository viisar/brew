from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.utils import check_X_y

from abc import abstractmethod


class DCS(object):

    def __init__(self, roc_selector=None, roc_size=7):
        if roc_selector is None:
            self.roc_selector = KNeighborsClassifier(n_neighbors=roc_size, algorithm='brute')
        elif hasattr(roc_selector, 'kneighbors'):
            self.roc_selector = roc_selector
        else:
            raise ValueError('roc_selector must implement kneighbors method')

        if roc_size < 1:
            raise ValueError('roc_size must be equal or greater than 1')
        else:
            self.roc_size = roc_size

    def select(self, ensemble, x):
        if len(x.shape) != 1:
            if x.shape[0] == 1:
                x = np.squeeze(x)
            else:
                raise ValueError('x must be one single sample at the time')

        if len(x) != self.Xval.shape[1]:
            raise ValueError('Sample x must have the same number of features'
                    'as the samples used to fit the selector (self.Xval)!')

        if ensemble.agrees(x.reshape(1,-1))[0]:
            selected_ensemble, weights = Ensemble(ensemble._estimators[0]), None
        else:
            selected_ensemble, weights = self._select(ensemble, x)

        return selected_ensemble, weights

    @abstractmethod
    def _select(self, ensemble, x):
        pass

    def fit(self, X, y):
        self.Xval, self.yval = check_X_y(X, y)
        return self

    def get_roc_idx(self, x, return_distance=False):
        if not return_distance:
            [idx] = self.roc_selector.kneighbors(x, return_distance=False)
        else:
            [dists], [idx] = self.knn.kneighbors(x, return_distance=True)

        if return_distance:
            return idx, dists
        else:
            return idx

    def get_roc(self, x, return_distance=False):
        idx, dists = self.get_roc_idx(x, return_distance=True)

        X_nn = self.Xval[idx]  # k neighbors
        y_nn = self.yval[idx]  # k neighbors target

        if return_distance:
            return X_nn, y_nn, dists
        else:
            return X_nn, y_nn

