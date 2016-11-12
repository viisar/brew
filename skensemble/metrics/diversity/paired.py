import numpy as np


def __coefficients(oracle):
    A = np.asarray(oracle[:, 0], dtype=bool)
    B = np.asarray(oracle[:, 1], dtype=bool)

    a = np.sum(A * B)           # A right, B right
    b = np.sum(~A * B)          # A wrong, B right
    c = np.sum(A * ~B)          # A right, B wrong
    d = np.sum(~A * ~B)         # A wrong, B wrong

    return a, b, c, d


def kuncheva_q_statistics(oracle):
    L = oracle.shape[1]
    div = np.zeros((L * (L - 1)) / 2)
    div_i = 0

    for i in range(L):
        for j in range(i + 1, L):
            a, b, c, d = __coefficients(oracle[:, [i, j]])
            div[div_i] = float(a * d - b * c) / ((a * d + b * c) + 10e-24)
            div_i = div_i + 1

    return np.mean(div)


def kuncheva_correlation_coefficient_p(oracle):
    L = oracle.shape[1]
    div = np.zeros((L * (L - 1)) / 2)
    div_i = 0

    for i in range(L):
        for j in range(i + 1, L):
            a, b, c, d = __coefficients(oracle[:, [i, j]])
            div[div_i] = float((a * d - b * c)) / \
                (np.sqrt((a + b) * (c + d) * (a + c) * (b + d)))
            div_i = div_i + 1

    return np.mean(div)


def kuncheva_disagreement_measure(oracle):
    L = oracle.shape[1]
    div = np.zeros((L * (L - 1)) / 2)
    div_i = 0

    for i in range(L):
        for j in range(i + 1, L):
            a, b, c, d = __coefficients(oracle[:, [i, j]])
            div[div_i] = float(b + c) / (a + b + c + d)
            div_i = div_i + 1

    return np.mean(div)


def kuncheva_agreement_measure(oracle):
    return 1.0 / (kuncheva_disagreement_measure(oracle) + 10e-24)


def kuncheva_double_fault_measure(oracle):
    L = oracle.shape[1]
    div = np.zeros((L * (L - 1)) / 2)
    div_i = 0

    for i in range(L):
        for j in range(i + 1, L):
            a, b, c, d = __coefficients(oracle[:, [i, j]])
            div[div_i] = float(d) / (a + b + c + d)
            div_i = div_i + 1

    return np.mean(div)


def __get_coefficients(y_true, y_pred_a, y_pred_b):
    a, b, c, d = 0, 0, 0, 0
    for i in range(y_true.shape[0]):
        if y_pred_a[i] == y_true[i] and y_pred_b[i] == y_true[i]:
            a = a + 1
        elif y_pred_a[i] != y_true[i] and y_pred_b[i] == y_true[i]:
            b = b + 1
        elif y_pred_a[i] == y_true[i] and y_pred_b[i] != y_true[i]:
            c = c + 1
        else:
            d = d + 1

    return a, b, c, d


def q_statistics(y_true, y_pred_a, y_pred_b):
    a, b, c, d = __get_coefficients(y_true, y_pred_a, y_pred_b)
    q = float(a * d - b * c) / (a * d + b * c)
    return q


def correlation_coefficient_p(y_true, y_pred_a, y_pred_b):
    a, b, c, d = __get_coefficients(y_true, y_pred_a, y_pred_b)
    p = float((a * d - b * c)) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return p


def disagreement_measure(y_true, y_pred_a, y_pred_b):
    a, b, c, d = __get_coefficients(y_true, y_pred_a, y_pred_b)
    disagreement = float(b + c) / (a + b + c + d)
    return disagreement


def agreement_measure(y_true, y_pred_a, y_pred_b):
    return 1.0 / disagreement_measure(y_true, y_pred_a, y_pred_b)


def double_fault_measure(y_true, y_pred_a, y_pred_b):
    a, b, c, d = __get_coefficients(y_true, y_pred_a, y_pred_b)
    df = float(d) / (a + b + c + d)
    return df


def paired_metric_ensemble(ensemble, X, y, paired_metric=q_statistics):
    classifiers = ensemble.classifiers
    size = len(classifiers)
    diversities = []
    for i in range(size):
        for j in range(i):
            y_pred_a = classifiers[i].predict(X)
            y_pred_b = classifiers[j].predict(X)
            diversity = paired_metric(y, y_pred_a, y_pred_b)
            diversities = diversities + [diversity]

    return np.mean(diversities)
