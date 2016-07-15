import numpy as np


def kuncheva_entropy_measure(oracle):
    L = oracle.shape[1]
    tmp = np.sum(oracle, axis=1)
    tmp = np.minimum(tmp, L - tmp)

    e = np.mean((1.0 / (L - np.ceil(0.5 * L))) * tmp)

    return e


def kuncheva_kw(oracle):
    L = oracle.shape[1]
    tmp = np.sum(oracle, axis=1)
    tmp = np.multiply(tmp, L - tmp)

    kw = np.mean((1.0 / (L**2)) * tmp)

    return kw


def new_entropy(ensemble, X, y):
    out = ensemble.output(X)
    P = np.sum(out, axis=2) / out.shape[1]
    P = - P * np.log(P + 10e-8)
    entropy = np.mean(np.sum(P, axis=1))

    return entropy


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

    e = (1.0 / len(X)) * (1.0 / (len(ensemble) - len(ensemble) / 2)) * factor

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

    kw = (1.0 / (len(X) * (len(ensemble)**2))) * factor
    return kw
