import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier

from .base import DCS

from brew.base import Ensemble
from brew.selection.dynamic.knora import KNORA
from brew.selection.dynamic.knora import KNORA_ELIMINATE
from brew.selection.dynamic.knora import KNORA_ELIMINATE_2
from brew.selection.dynamic.knora import KNORA_UNION
from brew.selection.dynamic.knora import KNORA_UNION_2
from brew.selection.dynamic.knora import _get_pool_mask

class KNORA_DB_U(KNORA):

    def is_indecision_region(self, x, ensemble, alpha=0.3):
        [dists], [idx] = self.knn.kneighbors(x, return_distance=True)
        y_nn = self.yval[idx] # k neighbors target

        if len(set(y_nn)) == 1:
            return False

        mx, mn = 0, len(y_nn)
        for y in set(y_nn):
            count = sum(y_nn == y)
            if d[y] < mn:
                mn = y
            if d[y] > mx:
                mx = y

        if mn <= alpha * mx:
            return False

        dist_lcl = sorted([float(d[k]) / np.sum(y_nn==k) for k in d.keys()], reverse=True)
        dist_sum = float(sum([v for (k, v) in d.items()]))
        if (max(dist_lcl)/dist_sum * alpha) >= (max(dist_lcl[1:])):
            return False

        return True

    def select(self, ensemble, x):
        neighbors_X, neighbors_y = self.get_neighbors(x)
        pool_output = ensemble.output(neighbors_X, mode='labels')

        output_mask = (pool_output == neighbors_y[:,np.newaxis])

        ensemble_mask = np.ones(len(ensemble), dtype=bool)

        
        for lbl in set(neighbors_y):
            np.multiply(pool_output == lbl, output_mask)
            lbl_mask = (neighbors_y == lbl)

            tmp = _get_pool_mask(output_mask[lbl_mask], neighbors_y[lbl_mask], np.any)
            tmp = 0 if tmp == None else tmp
            ensemble_mask = ensemble_mask * tmp

        [selected_idx] = np.where(ensemble_mask)

        if selected_idx.size > 0:
            if self.weighted:
                weights = 1.0/(np.sqrt(np.sum((x - neighbors_X)**2, axis=1)) + 10e-8)
                weighted_votes = np.dot(weights, output_mask[:,selected_idx])
            else:
                weighted_votes = np.sum(output_mask[:,selected_idx], axis=0)
        
            pool = Ensemble(classifiers=[ensemble.classifiers[i] for i in selected_idx])

        else: # if no classifiers are selected, use all classifiers with no weights
            knora_u = KNORA_UNION_2(self.Xval, self.yval, knn=self.knn, K = self.K)
            return knora_u.select(ensemble, x)
            #pool = ensemble
            #weighted_votes = None

        return pool, weighted_votes


class KNORA_DB_E(KNORA):

    def __init__(self, Xval, yval, K=5, weighted=False, knn=None, positive_class=1):
        self.Xval = Xval
        self.yval = yval
        self.K = K

        if knn == None:
            self.knn = KNeighborsClassifier(n_neighbors=K, algorithm='brute')
        else:
            self.knn = knn

        self.knn.fit(Xval, yval)
        self.weighted = weighted

        self.positive_class = positive_class



    def is_indecision_region(self, x, ensemble, alpha=0.2):
        [dists], [idx] = self.knn.kneighbors(x, return_distance=True)
        y_nn = self.yval[idx] # k neighbors target

        if len(set(y_nn)) == 1:
            return False

        d = {}
        for i, dist in zip(y_nn, dists):
            d[i] = d[i] + dist if i in d else dist

        dist_lcl = sorted([float(d[k]) / np.sum(y_nn==k) for k in d.keys()], reverse=True)
        dist_sum = float(sum([v for (k, v) in d.items()]))
        if (max(dist_lcl)/dist_sum * alpha) >= (max(dist_lcl[1:])):
            return False

        return True

    def select(self, ensemble, x):
        ensemble_mask = None
        
        neighbors_X, neighbors_y = self.get_neighbors(x)
        # pool_output (instances, classifiers)
        pool_output = ensemble.output(neighbors_X, mode='labels')


        #if len(set(neighbors_y)) == 1:
        if not self.is_indecision_region(x, ensemble):
            knora_e = KNORA_ELIMINATE_2(self.Xval, self.yval, 
                    K=self.K, weighted=False, knn=self.knn)
            selection, weights = knora_e.select(ensemble, x)
            knora_e = None
            return selection, weights

        pool_mask = _get_pool_mask(pool_output, neighbors_y, np.all)
        neighborhood_mask = np.ones(neighbors_y.shape[0], dtype=bool)
        if pool_mask is not None:
            ensemble_mask = pool_mask
        else:
            # gradually decrease neighborhood size if no
            # classifier predicts ALL the neighbors correctly
            # maintaining the decision region
            i = self.K - 1
            while i >= 0:
                neighborhood_mask[i] = False
                if set(neighbors_y[neighborhood_mask]) != set(neighbors_y):
                #if self.positive_class not in set(neighbors_y[neighborhood_mask]):
                    neighborhood_mask[i] = True
                else:
                    pool_mask = _get_pool_mask(pool_output[neighborhood_mask], 
                            neighbors_y[neighborhood_mask], np.all)
                    # if at least one classifier gets all neighbors right
                    if pool_mask is not None:
                        ensemble_mask = pool_mask
                        i = 0

                i = i - 1

                # if NO classifiers get the nearest neighbor correctly
        if ensemble_mask is None:
            # Increase neighborhood until one classifier
            # gets at least ONE (i.e. ANY) neighbors correctly. 
            # Starts with 2 because mask_all with k=1 is 
            # the same as mask_any with k=1
            knora_e = KNORA_ELIMINATE_2(self.Xval, self.yval, 
                    K=self.K, weighted=False, knn=self.knn)
            selection, weights = knora_e.select(ensemble, x)
            knora_e = None
            return selection, weights

            labels = set(neighbors_y)
            pool_mask = np.zeros(pool_output.shape[1])
            for i in range(-1, self.K):
                if i != -1 and neighborhood_mask[i] == False: 
                    neighborhood_mask[i] = True

                    for lbl in labels:
                        lbl_mask = (neighbors_y == lbl) * neighborhood_mask
                        tmp = _get_pool_mask(pool_output[lbl_mask], neighbors_y[lbl_mask], np.any)
                        tmp = 0 if tmp == None else tmp
                        pool_mask = pool_mask + tmp

                    pool_mask = pool_mask > 1
                    if np.any(pool_mask):
                        ensemble_mask = pool_mask
                        break

        [selected_idx] = np.where(ensemble_mask)

        if selected_idx.size > 0:
            pool = Ensemble(classifiers=[ensemble.classifiers[i] for i in selected_idx])
        else:
            pool = ensemble

        return pool, None


