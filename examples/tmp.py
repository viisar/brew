from brew.selection.dynamic.ola import OLA
from brew.selection.dynamic.ola import OLA2
from brew.selection.dynamic.lca import LCA
from brew.selection.dynamic.lca import LCA2

from brew.generation.bagging import Bagging

import numpy as np

import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split

N = 1000
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)

X, y = datasets.make_hastie_10_2(n_samples=N, random_state=1)
for i, yi in enumerate(set(y)):
    y[y == yi] = i

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30)

bag = Bagging(base_classifier=dt, n_classifiers=10)
bag.fit(X_val, y_val)

dcs_list = [OLA(X_val, y_val), OLA2(X_val, y_val), LCA(X_val, y_val), LCA2(X_val, y_val)]*100


import multiprocessing as mp
import time

a = time.time()
pool = mp.Pool(processes=4)

def function(dcs):
    y_pred = []
    for i in range(X_test.shape[0]):
        clf = dcs.select(bag.ensemble, X_test[i]).classifiers[0]
        y_pred = y_pred + [clf.predict(X_test[i])]
    return zero_one_loss(y_pred, y_test)

import multiprocessing as mp
import random
import string

# Define an output queue
output = mp.Queue()

# define a example function
def rand_string(dcs, output, pos):
    output.put((pos, function(dcs)))

# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(dcs, output, pos)) for pos, dcs in enumerate(dcs_list)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results_1 = [output.get() for p in processes]
results_1 = np.array(sorted(results_1))
results_1 = results_1[:,1]

#print(results_1)
print time.time() - a
a = time.time()
results_2 = []
for dcs in dcs_list:
    results_2 = results_2 + [function(dcs)]
#print results_2
print time.time() - a

results_1, results_2 = np.array(results_1), np.array(results_2)
print float(sum(results_1 == results_2))/results_1.shape[0]
    

