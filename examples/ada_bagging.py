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
from sklearn.cross_validation import train_test_split

from brew.generation.bagging import Bagging
from brew.generation.random_subspace import RandomSubspace
from brew.generation.random_newspace import RandomNewspace
from brew.generation.adaptive_bagging import AdaptiveBagging
from brew.combination.rules import majority_vote_rule
from brew.metrics.evaluation import auc_score


n_classifiers = 10
combination_rule='majority_vote'
max_samples=1.0
max_features=0.5
K=10
bootstrap_samples=0.75
bootstrap_features=0.75
n_components=1

X, y = datasets.make_hastie_10_2(n_samples=5000, random_state=1)
X = X[:,range(5)]
d = {}
for v, key in enumerate(set(y)):
    d[key] = v
y = np.asarray([d[yi] for yi in y])

mask = y == 1
X_1, X_0 = X[mask], X[~mask]
X_1 = X_1[:500]

X = np.concatenate([X_0,X_1], axis = 0)
y = np.array(2500*[0] + 500*[1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#X_test, y_test = X[:150], y[:150]
#X_train, y_train = X[150:], y[150:]

#dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
y_pred_tst = dt.predict(X_test)
dt_err = 1.0 - auc_score(y_test, y_pred_tst)

def ensemble_error(get_ensemble):
    error = np.zeros((n_classifiers,))
    for i in range(n_classifiers):
        if (i+1) % (n_classifiers/10) == 0:
            print i+1
        ensemble = get_ensemble(i+1)
        ensemble.fit(X_train, y_train)
        y_pred_tst = ensemble.predict(X_test)
        error[i] = 1.0 - auc_score(y_test, y_pred_tst)
    return error


print('running adaptive bagging')
a_bagging_error = ensemble_error(get_ensemble=lambda i: AdaptiveBagging(K=K, alpha=0.5, n_classifiers=i, base_classifier=dt))
print('running bagging')
bagging_error = ensemble_error(get_ensemble=lambda i: Bagging(base_classifier=dt, n_classifiers=i, combination_rule=combination_rule))
print('running random subspace')
r_subspace_error = ensemble_error(get_ensemble=lambda i: RandomSubspace(base_classifier=dt, n_classifiers=i, combination_rule=combination_rule, max_features=max_features))
print('running random newspace')
r_newspace_error = ensemble_error(get_ensemble=lambda i: RandomNewspace(base_classifier=dt, n_classifiers=i, combination_rule=combination_rule, K=K, bootstrap_samples=bootstrap_samples, bootstrap_features=bootstrap_features, max_samples=max_samples, max_features=max_features, n_components=n_components))
print('plotting graph')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_classifiers], [dt_err] * 2, 'k--',
        label='Decision Tree Error')

ax.plot(np.arange(n_classifiers) + 1, a_bagging_error,
        label='ABagging',
        color='black')
ax.plot(np.arange(n_classifiers) + 1, bagging_error,
        label='Bagging',
        color='red')
ax.plot(np.arange(n_classifiers) + 1, r_newspace_error,
        label='Random Newspace',
        color='blue')
ax.plot(np.arange(n_classifiers) + 1, r_subspace_error,
        label='Random Subspace',
        color='green')


ax.set_title('Comparision')
ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_classifiers')
ax.set_ylabel('error rate')

#leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = ax.legend(loc=4, scatterpoints=1, ncol=2)
#leg = ax.legend(loc='upper right', mode="expand", borderaxespad=0)
leg.get_frame().set_alpha(0.7)

plt.savefig('test.png')
print('done')
plt.show()
