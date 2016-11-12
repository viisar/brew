import numpy as np

from ..base import Ensemble
from ..combination.combiner import Combiner

from sklearn import cross_validation


class EnsembleStack(object):

    def __init__(self, mode='probs', cv=5):
        self.mode = mode
        self.layers = []
        self.cv = cv

    def add_layer(self, ensemble):
        if isinstance(ensemble, Ensemble):
            self.layers.append(ensemble)
        else:
            raise Exception('not an Ensemble object')

    def fit_layer(self, layer_idx, X, y):
        if layer_idx >= len(self.layers):
            return
        elif layer_idx == len(self.layers) - 1:
            self.layers[layer_idx].fit(X, y)
        else:
            n_classes = len(set(y)) - 1
            n_classifiers = len(self.layers[layer_idx])
            output = np.zeros((X.shape[0], n_classes * n_classifiers))
            skf = cross_validation.StratifiedKFold(y, self.cv)
            for tra, tst in skf:
                self.layers[layer_idx].fit(X[tra], y[tra])
                out = self.layers[layer_idx].output(X[tst], mode=self.mode)
                output[tst, :] = out[:, 1:, :].reshape(
                    out.shape[0], (out.shape[1] - 1) * out.shape[2])

            self.layers[layer_idx].fit(X, y)
            self.fit_layer(layer_idx + 1, output, y)

    def fit(self, X, y):
        if self.cv > 1:
            self.fit_layer(0, X, y)
        else:
            X_ = X
            for layer in self.layers:
                layer.fit(X_, y)
                out = layer.output(X_, mode=self.mode)
                X_ = out[:, 1:, :].reshape(
                    out.shape[0], (out.shape[1] - 1) * out.shape[2])

        return self

    def output(self, X):
        input_ = X

        for layer in self.layers:
            out = layer.output(input_, mode=self.mode)
            input_ = out[:, 1:, :].reshape(
                out.shape[0], (out.shape[1] - 1) * out.shape[2])

        return out


class EnsembleStackClassifier(object):

    def __init__(self, stack, combiner=None):
        self.stack = stack
        if combiner is None:
            self.combiner = Combiner(rule='mean')
        elif isinstance(combiner, str):
            if combiner == 'majority_vote':
                raise ValueError('EnsembleStackClassifier '
                        'do not support majority_vote')
            self.combiner = Combiner(rule=combiner)
        elif isinstance(combiner, Combiner):
            self.combiner = combiner
        else:
            raise ValueError('Invalid combiner!')


    def fit(self, X, y):
        self.stack.fit(X, y)
        return self
        
    def predict(self, X):
        out = self.stack.output(X)
        return self.combiner.combine(out)

    def predict_proba(self, X):
        out = self.stack.output(X)
        return np.mean(out, axis=2)
