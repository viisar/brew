import numpy as np

def entropy_measure_e(ensemble, X, y):
    factor = 0
    for j in range(y.shape[0]):
        right, wrong = 0, 0
        for estimator in ensemble.classifiers:
            [c] = estimator.predict(X[j])
            if c == y[j]:
                right = right + 1
            else:
                wrong = wrong + 1

        factor = factor + min(right, wrong)

    e = (1.0/len(X)) * (2.0/(len(ensemble) - 1)) * factor
    return e

