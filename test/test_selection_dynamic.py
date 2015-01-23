"""
Tests for `brew.base` module.  """

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from brew.base import Ensemble
from brew.generation.bagging import *
from brew.selection.dynamic.knora import *

Xtra = np.random.random((100, 2))
ytra = np.random.randint(0,2,100)

Xval = np.random.random((40,2))
yval = np.random.randint(0,2,40)

Xtst = np.random.random((30,2))
ytst = np.random.randint(0,2,30)

class TestKNORA():

    def test_simple(self):
        ensemble = Bagging(base_classifier=DecisionTreeClassifier(), n_classifiers=5)
        ensemble.fit(Xtra, ytra)

        selector = KNORA_E(Xval=Xval, yval=yval)

        pool = selector.select(ensemble, Xtst[0])

        print(pool)
        #print(pool)
