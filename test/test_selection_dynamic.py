"""
Tests for `brew.base` module.  """

import numpy as np

import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split

from brew.base import Ensemble
from brew.generation.bagging import *
from brew.selection.dynamic.knora import *

Xtra = np.random.random((100, 2))
ytra = np.random.randint(0,2,100)

Xval = np.random.random((40,2))
yval = np.random.randint(0,2,40)

Xtst = np.random.random((30,2))
ytst = np.random.randint(0,2,30)

N=1000
X, y = datasets.make_hastie_10_2(n_samples=N, random_state=1)
for i, yi in enumerate(set(y)):
    y[y == yi] = i

Xtra, Xtst, ytra, ytst = train_test_split(X, y, test_size=0.10)
Xtra, Xval, ytra, yval = train_test_split(Xtra, ytra, test_size=0.30)

bag = Bagging(base_classifier=DecisionTreeClassifier(), n_classifiers=5)
bag.fit(Xtra, ytra)

class TestKNORA_E():
    def test_simple(self):
        selector = KNORA_ELIMINATE(Xval=Xval, yval=yval)
        for x in Xtst:
            pool, w = selector.select(bag.ensemble, x)


class TestKNORA_U():

    def test_simple(self):
        selector = KNORA_UNION(Xval=Xval, yval=yval)
        for x in Xtst:
            pool, w = selector.select(bag.ensemble, x)

