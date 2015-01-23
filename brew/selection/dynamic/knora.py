import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier

from .base import DCS

from brew.base import Ensemble

def _get_pool_mask_eliminate(pool_output, neighbors_target):
    pool_mask = np.all(pool_output == neighbors_target[:,np.newaxis], axis=0)
    print('pool mask')
    print(pool_mask)

    # if at least one classifier gets all the neighbors right, return mask
    if np.sum(pool_mask) > 0:
        return pool_mask

    return None

def _get_pool_mask_union(pool_output, neighbors_target):
    pool_mask = np.any(pool_output == neighbors_target[:,np.newaxis], axis=0)
    print('pool mask')
    print(pool_mask)

    # if at least one classifier gets all the neighbors right, return mask
    if np.sum(pool_mask) > 0:
        return pool_mask

    return None


class KNORA_E(DCS):

    def select(self, ensemble, x):
        ensemble_mask = None
        k = self.K
      
        # obtain the K nearest neighbors in the validation set
        [idx] = self.knn.kneighbors(x, return_distance=False)
        neighbors_x = self.Xval[idx] # k neighbors
        neighbors_y = self.yval[idx] # k neighbors target

        # obtain pool output for the neighbors
        pool_output = np.zeros((neighbors_x.shape[0], len(ensemble)))
        for i, clf in enumerate(ensemble.classifiers):
            pool_output[:,i] = clf.predict(neighbors_x)

        # gradually decrease neighborhood size if no
        # classifier predicts all the neighbors correctly
        for i in range(k,0,-1):
            print('------- k = {} --------'.format(i))
            pool_mask = _get_pool_mask_eliminate(pool_output[:i], neighbors_y[:i])

            # if at least one classifier gets all neighbors right
            if pool_mask is not None:
                ensemble_mask = pool_mask
                break

        # if NO classifiers get the nearest neighbor correctly
        if ensemble_mask is None:
           
            # increase neighborhood until one classifier
            # gets at least ONE neighbor correctly
            #
            # starts with 2 because mask_union with k=1 is 
            # the same as mask_eliminate with k=1
            for i in range(2,k+1):
                pool_mask = _get_pool_mask_union(pool_output[:i], neighbors_y[:i])

                if pool_mask is not None:
                    ensemble_mask = pool_mask
                    break

        [selected_idx] = np.where(ensemble_mask)
        pool = [ensemble.classifiers[i] for i in selected_idx]

        return Ensemble(classifiers=pool)


