# -*- coding: utf-8 -*-
import numpy as np

from .base import Prunner


class EPIC(Prunner):

    def __init__(self):
        pass

    def fit(self, ensemble, X, y):
        classifiers = ensemble.classifiers
        self.classes_ = list(set(y))
        V = np.zeros((y.shape[0], len(self.classes_)))
        C = classifiers

        index_table = {}
        i = 0
        for classe in self.classes_:
            index_table[classe] = i
            i = i + 1

        for c_i in C:
            for i in range(y.shape[0]):
                prediction = c_i.predict(X[i])[0]
                index = index_table[prediction]
                V[i][index] = V[i][index] + 1

        OL = []
        for c_i in C:
            IC = 0.0
            for j in range(y.shape[0]):
                pred = c_i.predict(X[j])[0]
                index_pred = index_table[pred]

                alpha, beta, gamma = 0, 0, 1
                if index_pred == index_table[y[j]]:
                    alpha = 1 if index_pred == np.argmin(V[j]) else 0
                    beta = 1 if index_pred == np.argmax(V[j]) else 0
                    gamma = 0

                v_max = np.argmax(V[j])
                v_sec = sorted(np.array(V[j]))[len(V[j]) - 2]
                v_cor = V[j][index_table[y[j]]]

                IC = IC + alpha * \
                    (2 * v_max - V[j][index_pred]) + beta * v_sec + \
                    gamma * (v_cor - V[j][index_pred] - v_max)

            OL = OL + [[c_i, IC]]

        OL = sorted(OL, key=lambda e: e[1], reverse=True)
        self.classifiers = list(zip(*OL)[0])
        return self

    def get(self, p=0.1):
        return self.classifiers[:int(p * len(self.classifiers))]
