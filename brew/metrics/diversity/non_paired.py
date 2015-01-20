import numpy as np

def entropy_measure_e(ensemble, X, y):
    factor = 0
    print len(ensemble)
    for j in range(y.shape[0]):
        right, wrong = 0, 0
        for estimator in ensemble.classifiers:
            [c] = estimator.predict(X[j])
            if c == y[j]:
                right = right + 1
            else:
                wrong = wrong + 1

        factor = factor + min(right, wrong)

    e = (1.0/len(X)) * (1.0 / (len(ensemble) - len(ensemble)/2)) * factor
    
    return e

def kohavi_wolpert_variance(ensemble, X, y):
    factor = 0
    for j in range(y.shape[0]):
        right, wrong = 0, 0
        for estimator in ensemble.classifiers:
            [c] = estimator.predict(X[j])
            if c == y[j]:
                right = right + 1
            else:
                wrong = wrong + 1

        factor = factor + right * wrong

    kw =  (1.0/(len(X)*(len(ensemble)**2))) * factor
    return kw

