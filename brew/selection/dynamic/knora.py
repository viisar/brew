import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier

from .base import DCS

from brew.base import Ensemble


# do not use this class directly, call it's subclasses instead (e.g. KNORA_E)
class KNORA(DCS):

    def get_neighbors(self, x):
        # obtain the K nearest neighbors of test sample in the validation set
        [idx] = self.knn.kneighbors(x, return_distance=False)
        X_nn = self.Xval[idx] # k neighbors
        y_nn = self.yval[idx] # k neighbors target

        return X_nn, y_nn

    
class KNORA_ELIMINATE(KNORA):

    def select(self, ensemble, x):
        ensemble_mask = None

        neighbors_X, neighbors_y = self.get_neighbors(x)
        pool_output = ensemble.output(neighbors_X, mode='labels')

        # gradually decrease neighborhood size if no
        # classifier predicts ALL the neighbors correctly
        for i in range(self.K, 0, -1):
            pool_mask = _get_pool_mask(pool_output[:i], neighbors_y[:i], np.all)

            # if at least one classifier gets all neighbors right
            if pool_mask is not None:
                ensemble_mask = pool_mask
                break

        # if NO classifiers get the nearest neighbor correctly
        if ensemble_mask is None:
           
            # Increase neighborhood until one classifier
            # gets at least ONE (i.e. ANY) neighbors correctly. 
            # Starts with 2 because mask_all with k=1 is 
            # the same as mask_any with k=1
            for i in range(2, self.K+1):
                pool_mask = _get_pool_mask(pool_output[:i], neighbors_y[:i], np.any)

                if pool_mask is not None:
                    ensemble_mask = pool_mask
                    break

        [selected_idx] = np.where(ensemble_mask)

        if selected_idx.size > 0:
            pool = Ensemble(classifiers=[ensemble.classifiers[i] for i in selected_idx])

        else: # use all classifiers
            pool = ensemble


        # KNORA-ELIMINATE-W that supposedly uses weights, does not make
        # any sense, so even if self.weighted is True, always return
        # None for the weights

        return pool, None


class KNORA_UNION(KNORA):

    def select(self, ensemble, x):
        neighbors_X, neighbors_y = self.get_neighbors(x)
        pool_output = ensemble.output(neighbors_X, mode='labels')

        output_mask = (pool_output == neighbors_y[:,np.newaxis])

        [selected_idx] = np.where(np.any(output_mask, axis=0))

        if selected_idx.size > 0:
            if self.weighted:
                weights = np.sqrt(np.sum((x - neighbors_X)**2, axis=1))
                weighted_votes = np.dot(weights, output_mask[:,selected_idx])
            else:
                weighted_votes = np.sum(output_mask[:,selected_idx], axis=0)
        
            pool = Ensemble(classifiers=[ensemble.classifiers[i] for i in selected_idx])

        else: # if no classifiers are selected, use all classifiers with no weights
            pool = ensemble
            weighted_votes = None

        return pool, weighted_votes


class KNORA_E_DB(KNORA):

     def select(self, ensemble, x):
        ensemble_mask = None

        neighbors_X, neighbors_y = self.get_neighbors(x)
        pool_output = ensemble.output_simple(neighbors_X)

        if len(set(neighbors_y)) == 1:
            knora_e = KNORA_ELIMINATE(self.Xval, self.yval, 
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
            i = self.K - 1
            while i >= 0:
                neighborhood_mask[i] = False
                if set(neighbors_y[neighborhood_mask]) != set(neighborhood_mask):
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
            labels = set(neighbors_y)
            pool_mask = np.zeros(pool_output.shape[1])
            for i in range(-1, self.K+1):
                if i == -1 or neighborhood_mask[i] == False: 
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

        pool = [ensemble.classifiers[i] for i in selected_idx]

        return Ensemble(classifiers=pool)


def _get_pool_mask(pool_output, neighbors_target, func):
    pool_mask = func(pool_output == neighbors_target[:,np.newaxis], axis=0)

    if np.sum(pool_mask) > 0:
        return pool_mask

    return None
