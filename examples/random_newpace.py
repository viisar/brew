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
from brew.combination.rules import majority_vote_rule


n_classifiers = 10
K=10
bootstrap_samples=0.75
bootstrap_features=0.75
max_samples=0.75
max_features=0.75
combination_rule=majority_vote_rule

parameters_list = [
        [2,0.75,0.5,0.3,0.6, 1],
        [2,0.75,0.5,0.3,0.6, 3],
        [2,0.75,0.5,0.3,0.6, 6],
        [2,0.75,0.5,0.3,0.6, 10],
        ]
'''
parameters_list = [
        [2,0.75,0.75,0.75,0.75, 100],
        [2,0.5,0.5,0.5,0.5, 100],
        [2,0.3,0.3,0.3,0.3, 100],
        [2,0.75,0.5,0.3,0.6, 100]
        ]
'''     

X, y = datasets.make_hastie_10_2(n_samples=5000, random_state=1)
#X = X[:,[0,1,2,3,4,5]]
d = {}
for v, key in enumerate(set(y)):
    d[key] = v
y = np.asarray([d[yi] for yi in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#X_test, y_test = X[:150], y[:150]
#X_train, y_train = X[150:], y[150:]

#dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)

def ensemble_error(get_ensemble):
    error = np.zeros((n_classifiers,))
    for i in range(n_classifiers):
        if (i+1) % (n_classifiers/10) == 0:
            print i+1
        ensemble = get_ensemble(i+1)
        ensemble.fit(X_train, y_train)
        y_pred_tst = ensemble.predict(X_test)
        error[i] = zero_one_loss(y_pred_tst, y_test)
    return error


errors = []
for ps in parameters_list:
    print('running random newspace')
    error = ensemble_error(get_ensemble=lambda i: RandomNewspace(base_classifier=dt, n_classifiers=i, combination_rule=combination_rule, K=ps[0], bootstrap_samples=ps[1], bootstrap_features=ps[2], max_samples=ps[3], max_features=ps[4], n_components=ps[5]))
    errors += [error]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_classifiers], [dt_err] * 2, 'k--',
        label='Decision Tree Error')

colors = ['red', 'blue', 'green', 'yellow']
for i, (error, color) in enumerate(zip(errors, colors)):
    ax.plot(np.arange(n_classifiers) + 1, error,
            label='Random Newspace K=' + str(parameters_list[i][0]),
            color=color)

ax.set_title('Comparision')
ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_classifiers')
ax.set_ylabel('error rate')

#leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = ax.legend(loc=1, scatterpoints=1, ncol=2)
#leg = ax.legend(loc='upper right', mode="expand", borderaxespad=0)
leg.get_frame().set_alpha(0.7)
plt.savefig('rs.png')
print('done')
plt.show()
