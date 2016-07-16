import numpy as np

from abc import abstractmethod

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from collections import Counter

class RegionOfCompetenceSelector(object):

    def __init__(self, roc_sel_type='neighbors', n_neighbors=7, radius=1.0, **kwargs):
        roc_sel_type_list = ('neighbors', 'radius', 'custom')
        if roc_sel_type not in roc_sel_type_list:
            raise NotImplementedError

        if roc_sel_type == 'custom':
            self.estimator = kwargs['estimator']
        else:
            self.roc_sel_type = roc_sel_type
            self.estimator = NearestNeighbors(n_neighbors, **kwargs)

        self.n_neighbors = n_neighbors
        self.radius = radius


    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X = X
        self.y = y
        self.stats_c_ = Counter(y)
        return self


    def _check_input(x):
        if not hasattr(self, 'stats_c_'):
            raise RuntimeError('You need to fit the validation set first!')

        if len(x.shape) > 2:
            raise ValueError('Invalid input shape!')

        if len(x.shape) == 1:
            x_ = x.reshape(-1,1)

        if x_.shape[1] != self.X.shape[1]:
            raise RuntimeError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.X.shape[1], x_.shape[1]))
        return x_


    def region_of_competence_index(self, x, delimiter=None, return_distance=False):
        if delimiter is None:
            delimiter = self.radius if self.roc_sel_type == 'radius' else self.n_neighbors

        x = _check_input(x)

        if self.roc_sel_type == 'neighbors':
            out = self.roc_sel.kneighbors(x, X=self.X, n_neighbors=delimiter, return_distance=return_distance)
        elif self.roc_sel_type == 'radius':
            out = self.roc_sel.radius_neighbors(x, X=self.X, radius=delimiter, return_distance=return_distance)

        if return_distance:
            [dists], [idx] = out
        else:
            [idx] = out

        if return_distance:
            return idx, dists
        else:
            return idx


    def region_of_competence(self, x, delimiter=None, return_distance=False):
        out = self.region_of_competence_index(x, delimiter=delimiter, return_distance=return_distance)

        if return_distance:
            dists, idx = out
            return self.X[idx], self.y[idx], dists
        else:
            idx = out
            return self.X[idx], self.y[idx]

