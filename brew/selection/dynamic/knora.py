import numpy as np

from sklearn.ensemble import BaggingClassifier
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


class KNORA_ELIMINATE_2(KNORA):
    def select(self, ensemble, x):
        neighbors_X, neighbors_y = self.get_neighbors(x)

        k = self.K

        pool = []
        while k > 0:
            nn_X = neighbors_X[:k,:]
            nn_y = neighbors_y[:k]

            for i, c in enumerate(ensemble.classifiers):
                if np.all(c.predict(nn_X) == nn_y[np.newaxis, :]):
                    pool.append(c)

            if not pool: # empty
                k = k-1
            else:
                break

        if not pool: # still empty
            # select the classifier that recognizes
            # more samples in the whole neighborhood
            # also select classifiers that recognize
            # the same number of neighbors
            pool = self._get_best_classifiers(ensemble, neighbors_X, neighbors_y, x)


        return Ensemble(classifiers=pool), None


    def _get_best_classifiers(self, ensemble, neighbors_X, neighbors_y, x):
        ensemble_out = ensemble.output(neighbors_X, mode='labels')
        ensemble_mask = ensemble_out == neighbors_y[:,np.newaxis]

        correct = np.sum(ensemble_mask, axis=0)
        idx = np.argmax(correct) # best classifier idx

        all_idx = correct == correct[idx]
        
        pool = [ensemble.classifiers[i] for i in all_idx]

        return pool


class KNORA_UNION(KNORA):

    def select(self, ensemble, x):
        neighbors_X, neighbors_y = self.get_neighbors(x)
        pool_output = ensemble.output(neighbors_X, mode='labels')

        output_mask = (pool_output == neighbors_y[:,np.newaxis])

        [selected_idx] = np.where(np.any(output_mask, axis=0))

        if selected_idx.size > 0:
            if self.weighted:
                weights = 1.0/(np.sqrt(np.sum((x - neighbors_X)**2, axis=1)) + 10e-8)
                weighted_votes = np.dot(weights, output_mask[:,selected_idx])
            else:
                weighted_votes = np.sum(output_mask[:,selected_idx], axis=0)
        
            pool = Ensemble(classifiers=[ensemble.classifiers[i] for i in selected_idx])

        else: # if no classifiers are selected, use all classifiers with no weights
            pool = ensemble
            weighted_votes = None

        return pool, weighted_votes

class KNORA_UNION_2(KNORA):
    
    def select(self, ensemble, x):
        neighbors_X, neighbors_y = self.get_neighbors(x)
       
        pool = []
        for i, neighbor in enumerate(neighbors_X):
            for c in ensemble.classifiers:
                if c.predict(neighbor) == neighbors_y[i]:
                    pool.append(c)

        return Ensemble(classifiers=pool), None


def _get_pool_mask(pool_output, neighbors_target, func):
    pool_mask = func(pool_output == neighbors_target[:,np.newaxis], axis=0)

    if np.sum(pool_mask) > 0:
        return pool_mask

    return None
