"""
Tests for `brew.selection.dynamic` module.  """

import numpy as np

import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split

from brew.base import Ensemble
from brew.generation.bagging import *
from brew.selection.dynamic.knora import *
from brew.selection.dynamic.lca import LCA, LCA2

N=100
X, y = datasets.make_hastie_10_2(n_samples=N, random_state=1)
for i, yi in enumerate(set(y)):
    y[y == yi] = i

Xtra, Xtst, ytra, ytst = train_test_split(X, y, test_size=0.10)
Xtra, Xval, ytra, yval = train_test_split(Xtra, ytra, test_size=0.30)

bag = Bagging(base_classifier=DecisionTreeClassifier(), n_classifiers=100)
bag.fit(Xtra, ytra)

class KNORA_UNION_VALID(KNORA):
    
    def select(self, ensemble, x):
        neighbors_X, neighbors_y = self.get_neighbors(x)
       
        pool = []
        for c in ensemble.classifiers:
            for i, neighbor in enumerate(neighbors_X):
                if c.predict(neighbor) == neighbors_y[i]:
                    pool.append(c)
                    break

        weights = []
        for clf in pool:
            msk = clf.predict(neighbors_X) == neighbors_y
            weights = weights + [sum(msk)]

        return Ensemble(classifiers=pool), weights



class KNORA_ELIMINATE_VALID(KNORA):
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

class TestKNORA_E():
    def test_simple(self):
        selector_pred = KNORA_ELIMINATE(Xval=Xval, yval=yval)
        selector_true = KNORA_ELIMINATE_VALID(Xval=Xval, yval=yval)
        for x in Xtst:
            pool_pred, w_pred = selector_pred.select(bag.ensemble, x)
            pool_true, w_true = selector_true.select(bag.ensemble, x)
            assert w_pred == w_true
            assert len(pool_pred) == len(pool_true)
            for c_p, c_t in zip(pool_pred.classifiers, pool_true.classifiers):
                assert c_p == c_t


class TestKNORA_U():

    def test_simple(self):
        selector_pred = KNORA_UNION(Xval=Xval, yval=yval)
        selector_true = KNORA_UNION_VALID(Xval=Xval, yval=yval)
        for x in Xtst:
            pool_pred, w_pred = selector_pred.select(bag.ensemble, x)
            pool_true, w_true = selector_true.select(bag.ensemble, x)
            assert len(pool_pred) == len(pool_true)
            for c_p, c_t in zip(pool_pred.classifiers, pool_true.classifiers):
                assert c_p == c_t
            assert len(w_pred) == len(w_true)
            assert np.all(np.array(w_pred) == np.array(w_true))

class TestLCA():
    def test_simple(self):
        selector_pred = LCA(Xval=Xval, yval=yval)
        for x in Xtst:
            pool_pred, w_pred = selector_pred.select(bag.ensemble, x)
            assert w_pred is None

    def test_simple2(self):
        selector_pred = LCA2(Xval=Xval, yval=yval)
        for x in Xtst:
            pool_pred, w_pred = selector_pred.select(bag.ensemble, x)
            assert w_pred is None


