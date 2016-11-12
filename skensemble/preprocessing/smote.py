from __future__ import division

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(T, N=100, k=1):
    """
    T: minority class data
    N: percentage of oversampling
    k: number of neighbors used
    """

    # modification of original smote code so that it won't break if
    # minority class is too small in relation to the k, maybe this is not
    # sensible.
    if T.shape[0] <= k + 1:
        idx = np.random.choice(T.shape[0], size=(k + 1,))
        T = T[idx, :]

    # randomly select a subset of the data, to be used for creating synthethic
    # samples
    if N < 100:
        sz = int(T.shape[0] * (N / 100))
        idx = np.random.choice(T.shape[0], size=(sz,), replace=False)
        T = T[idx, :]
        N = 100

    if N % 100 != 0:
        raise ValueError('N must be < 100 OR multiple of 100')

    N = int(N / 100)
    n_minority_samples, n_features = T.shape
    n_synthetic_samples = N * n_minority_samples

    synthetic = np.zeros((n_synthetic_samples, n_features))

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(T)

    count = 0

    for i in range(n_minority_samples):
        # first neighbor returned is always the very own sample, so
        # get 1 more neighbor and discard the first neighbor returned
        neighbors_idx = knn.kneighbors(
            T[i, :].reshape(1,-1), n_neighbors=k + 1, 
            return_distance=False)[0][1:]

        # randomly choose N neighbors of the sample (with replacement)
        nn_idx = np.random.choice(neighbors_idx, size=(N,))
        chosen_neighbors = T[nn_idx, :]

        diff = chosen_neighbors - T[i, :]
        gap = np.random.uniform(low=0.0, high=1.0, size=N)[:, np.newaxis]

        synthetic[count:count + N, :] = T[i, :] + (gap * diff)
        count += N

    return synthetic
