"""
Random Subspace 
"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris

from skensemble.generation.random_subspace import RandomSubspace

dtc = DecisionTreeClassifier()
dtr = DecisionTreeRegressor()

iris = load_iris()
X, y = iris['data'], iris['target']
del iris

rs = RandomSubspace(base_estimator=DecisionTreeClassifier(), n_estimators=123)
rs.fit(X, y)
ensemble = rs.ensemble
assert len(ensemble) == n_estimators
assert set(y) == set(ensemble.classes_)


