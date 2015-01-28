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
        pool_output = ensemble.output_simple(neighbors_X)

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

        print(ensemble_mask)

        pool = [ensemble.classifiers[i] for i in selected_idx]

        # KNORA-ELIMINATE-W that supposedly uses weights, does not make
        # any sense, so even if self.weighted is True, do nothing and
        # return None as weight vector

        return Ensemble(classifiers=pool), None


class KNORA_UNION(KNORA):

    def select(self, ensemble, x):
        neighbors_X, neighbors_y = self.get_neighbors(x)
        pool_output = ensemble.output_simple(neighbors_X)

        output_mask = (pool_output == neighbors_y[:,np.newaxis])

        [selected_idx] = np.where(np.any(output_mask, axis=0))

        print(output_mask[:,selected_idx])
       
        if self.weighted:
            weights = np.sqrt(np.sum((x - neighbors_X)**2, axis=1))
            weighted_votes = np.dot(weights, output_mask[:,selected_idx])
        else:
            weighted_votes = np.sum(output_mask[:,selected_idx], axis=0)
              
        pool = [ensemble.classifiers[i] for i in selected_idx]

        assert len(pool) == weighted_votes.size

        return Ensemble(classifiers=pool), weighted_votes


def _get_pool_mask(pool_output, neighbors_target, func):
    pool_mask = func(pool_output == neighbors_target[:,np.newaxis], axis=0)

    if np.sum(pool_mask) > 0:
        return pool_mask

    return None
