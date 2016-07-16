import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn.utils import check_X_y

from .region_of_competence import RegionOfCompetenceSelector

class DCS(ABCMeta):

    def __init__(self, X_val=None, y_val=None, ensemble=None, roc_sel_type='neighbors', n_neighbors=7, radius=1.0, **kwargs):
        self.ensemble = ensemble

        roc_sel_type_list = ('neighbors', 'radius', 'custom')
        if roc_sel_type not in roc_sel_type_list:
            raise NotImplementedError

        if roc_sel_type == 'custom':
            if 'roc_sel' in kwargs:
                roc_sel = kwargs['roc_sel']
            else:
                raise Exception('roc_sel_type=\'custom\' requires a \'roc_sel\' argument')
        else:
            #self.roc_sel = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, **kwargs)
            self.roc_sel = RegionOfCompetenceSelector(roc_sel_type=roc_sel_type,
                    n_neighbors=n_neighbors, radius=radius, **kwargs)

        self.n_neighbors = n_neighbors
        self.radius = radius

        if X_val is not None and y_val is not None:
            self.fit(X_val, y_val)


    def fit(self, X_val, y_val):
        X, y = check_X_y(X, y)
        self.roc_sel.fit(X_val, y_val)

    def select(self, x):
        # Check if roc_sel has been fitted
        pass

    @abstractmethod
    def _select(self, x):
        pass

    def region_of_competence(self, x, delimiter=None, return_distance=False):
        # obtain the K nearest neighbors of test sample in the validation set
        return self.roc_sel.region_of_competence(x, delimiter=delimiter, return_distance=return_distance)

    def region_of_competence_idx(self, x, delimiter=None, return_distance=False):
        # obtain the K nearest neighbors of test sample in the validation set
        return self.roc_sel.region_of_competence_idx(x, delimiter=delimiter, return_distance=return_distance)

