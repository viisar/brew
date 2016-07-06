import numpy as np

import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split

from brew.generation.bagging import Bagging
from brew.base import Ensemble, EnsembleClassifier
from brew.selection.pruning.epic import EPIC

N = 1000
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)

X, y = datasets.make_hastie_10_2(n_samples=N, random_state=1)
for i, yi in enumerate(set(y)):
    y[y == yi] = i

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30)

bag = Bagging(base_classifier=dt, n_classifiers=10)
bag.fit(X_val, y_val)

epic = EPIC()
epic.fit(bag.ensemble, X_test, y_test)

print('-----------------ERROR RATE----------------------')
for p in np.arange(0.1,1.1,0.1):
    ensemble = epic.get(p)
    mcs = EnsembleClassifier(Ensemble(classifiers=epic.get(p)), selector=None)
    y_pred = mcs.predict(X_test)
    print('p={}, {}'.format(p, zero_one_loss(y_pred, y_test)))
print ('------------------------------------------------')
