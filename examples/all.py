"""
=============================
Pool Generation Modules
=============================

This example is based on Figure 10.2 from Hastie et al 2009 [1] and illustrates
the difference in performance between the pool generation algorithms.

"""
print(__doc__)

# Author: Dayvid Victor <victor.dvro@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

from brew.generation.bagging import Bagging
from brew.generation.random_subspace import RandomSubspace
from brew.generation.random_newspace import RandomNewspace


n_classifiers = 100
combination_rule=majority_vote_rule
max_samples=0.75
max_features=0.5
K=10
bootstrap_samples=0.75
bootstrap_features=0.75



X, y = datasets.make_hastie_10_2(n_samples=5000, random_state=1)
X_test, y_test = X[:1500], y[:1500]
X_train, y_train = X[1500:], y[1500:]

#dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)

def ensemble_error(get_ensemble):
    error = np.zeros((n_classifiers,))
    for i in range(n_classifiers):
        ensemble = get_ensemble(i)
        ensemble.fit(X_train, y_train)
        y_pred_tst = ensemble.predict(X_test)
        error[i] = zero_one_loss(y_pred_tst, y_test)
    return error


bagging_error = ensemble_error(get_ensemble=lambda i: Bagging(base_classifier=dt, n_classifiers=i, combination_rule=combination_rule))
r_subspace_error = ensemble_error(get_ensemble=lambda i: RandomSubspace(base_classifier=dt, n_classifiers=i, combination_rule=combination_rule, max_features=max_features))
r_newspace_error = ensemble_error(get_ensemble=lambda i: RandomNewspace(base_classifier=dt, n_classifiers=i, combination_rule=combination_rule, K=K, bootstrap_samples=bootstrap_samples, bootstrap_features=bootstrap_features, max_samples=max_samples, max_features=max_features))


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_classifiers], [dt_err] * 2, 'k--',
        label='Decision Tree Error')

ax.plot(np.arange(n_classifiers) + 1, bagging_error,
        label='Bagging',
        color='red')

ax.plot(np.arange(n_classifiers) + 1, r_subspace_error,
        label='Random Subspace',
        color='blue')
ax.plot(np.arange(n_classifiers) + 1, r_newspace_error,
        label='Random Newspace',
        color='green')

ax.set_title('Comparision')
ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_classifiers')
ax.set_ylabel('error rate')

#leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = ax.legend(loc=1, scatterpoints=1, ncol=2)
#leg = ax.legend(loc='upper right', mode="expand", borderaxespad=0)
leg.get_frame().set_alpha(0.7)

plt.savefig('test.png')
plt.show()
