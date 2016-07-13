import numpy as np

from abc import abstractmethod
from sklearn.neighbors import NearestNeighbors

from .region_of_competence import RegionOfCompetenceSelector

class DCS(object):

    @abstractmethod
    def select(self, ensemble, x):
        pass

    def __init__(self, X_val, y_val, roc_sel_type='neighbors', n_neighbors=7, radius=1.0, **kwargs):
        self.X_val = X_val
        self.y_val = y_val

        roc_sel_type_list = ('neighbors', 'radius', 'custom')
        if roc_sel_type not in roc_sel_type_list:
            raise NotImplementedError

        if roc_sel_type == 'custom':
            if 'roc_sel' in kwargs:
                roc_sel = kwargs['roc_sel']
            else:
                raise Exception('roc_sel_type=\'custom\' requires a \'roc_sel\' argument')
        else:
            self.roc_sel = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, **kwargs)

        self.n_neighbors = n_neighbors
        self.radius = radius

    def region_of_competence(self, x, delimiter=None, return_distance=False):
        # obtain the K nearest neighbors of test sample in the validation set
        if delimiter is None:
            delimiter = self.radius if self.roc_sel_type == 'radius' else self.n_neighbors

        if self.roc_sel_type == 'neighbors':
            out = self.roc_sel.kneighbors(x.reshape(-1,1), n_neighbors=delimiter, return_distance=return_distance)
        elif self.roc_sel_type == 'radius':
            out = self.roc_sel.radius_neighbors(x.reshape(-1,1), radius=delimiter, return_distance=return_distance)

        if return_distance:
            [dists], [idx] = out
        else:
            [idx] = out

        X_nn = self.X_val[idx] # k neighbors
        y_nn = self.y_val[idx] # k neighbors target

        if return_distance:
            return X_nn, y_nn, dists
        else:
            return X_nn, y_nn

    def region_of_competence_idx(self, x, delimiter=None):
        # obtain the K nearest neighbors of test sample in the validation set
        if delimiter is None:
            delimiter = self.radius if self.roc_sel_type == 'radius' else self.n_neighbors

        if self.roc_sel_type == 'neighbors':
            [idx] = self.roc_sel.kneighbors(x.reshape(-1,1), n_neighbors=delimiter, return_distance=False)
        elif self.roc_sel_type == 'radius':
            [idx] = self.roc_sel.radius_neighbors(x.reshape(-1,1), radius=delimiter, return_distance=False)


        return idx

