import numpy as np
import matplotlib.pyplot as plt
from brew.preprocessing.smote import *


X = np.array([ 
    [7,4],
    [5,3],
    [2,8]
])

y = np.array([0,0,0])

N = 100
k = 1

rng = np.random.RandomState(123)

S = smote(X, N, k)



#X = np.concatenate([X, a], axis=0)
#y = np.concatenate([y, np.array([1,1,1])], axis=0)

a = plt.scatter(X[:,0], X[:,1], c='b')
b = plt.scatter(S[:,0], S[:,1], c='g')


plt.legend((a, b),
           ('Minority Class', 'Synthetic Samples'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=12)


plt.show()
