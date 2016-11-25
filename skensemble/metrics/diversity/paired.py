import numpy as np

def __coefficients(oracle):
    """Diversity Coefficients"""
    A = np.asarray(oracle[:, 0], dtype=bool)
    B = np.asarray(oracle[:, 1], dtype=bool)

    a = np.sum(A * B)           # A right, B right
    b = np.sum(~A * B)          # A wrong, B right
    c = np.sum(A * ~B)          # A right, B wrong
    d = np.sum(~A * ~B)         # A wrong, B wrong

    return a, b, c, d

def q_statistics(oracle):
    """Q Statistics"""
    L = oracle.shape[1]
    div = np.zeros((L * (L - 1)) / 2)
    div_i = 0

    for i in range(L):
        for j in range(i + 1, L):
            a, b, c, d = __coefficients(oracle[:, [i, j]])
            div[div_i] = float(a * d - b * c) / ((a * d + b * c) + 10e-24)
            div_i = div_i + 1

    return np.mean(div)

def correlation_coefficient_rho(oracle):
    """Correlation Coefficient Rho"""
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

def disagreement(oracle):
    """Disagreement Measure"""
    L = oracle.shape[1]
    div = np.zeros((L * (L - 1)) / 2)
    div_i = 0

    for i in range(L):
        for j in range(i + 1, L):
            a, b, c, d = __coefficients(oracle[:, [i, j]])
            div[div_i] = float(b + c) / (a + b + c + d)
            div_i = div_i + 1

    return np.mean(div)

def agreement(oracle):
    """Agreement Measure"""
    return 1.0 / (disagreement(oracle) + 10e-24)

def double_fault(oracle):
    """Double Fault Measure"""
    L = oracle.shape[1]
    div = np.zeros((L * (L - 1)) / 2)
    div_i = 0

    for i in range(L):
        for j in range(i + 1, L):
            a, b, c, d = __coefficients(oracle[:, [i, j]])
            div[div_i] = float(d) / (a + b + c + d)
            div_i = div_i + 1

    return np.mean(div)

