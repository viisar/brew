import numpy as np

from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_array, check_X_y

from skensemble.combination import Combiner
from skensemble.ensemble import Ensemble, is_ensemble_of_classifiers, is_ensemble_of_regressors

class EnsembleEstimatorMixin(object):
    pass

class BaseEnsembleEstimator(object):
    pass

class EnsembleClassifier(object):

    def __init__(self, ensemble=None, selector=None, combiner='majority_vote'):

        if ensemble is None or not is_ensemble_of_classifiers(ensemble):
            raise ValueError('Estimators in ensemble must be classifiers!')

        self.ensemble = ensemble

        #TODO validate selector
        self.selector = selector

        if isinstance(combiner, str):
            self.combiner = Combiner(rule=combiner)
        elif isinstance(combiner, Combiner):
            self.combiner = combiner
        else:
            raise ValueError('Invalid parameter combiner')

    @property
    def classes_(self):
        retur self.ensemble.classes_

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.ensemble.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)

        y_pred = np.zeros_line(X[:,0], dtype=int)

        classes_ = self.ensemble.classes_

        for idx in range(X.shape[0]):
            if self.selector is not None:
                ensemble, weights = self.selector.select(self.ensemble, X[idx,:])
                if ensemble.classes_ != classes_:
                    raise Exception('Selecting subset of classifiers with'
                            'different classes than the original ensemble!'
                            'The selector requires all classifiers to be'
                            'trained with the same classes!')

            else:
                ensemble, weights = self.ensemble, None

            ensemble_output = ensemble.output(X[idx,:].reshape(1,-1))
            y_pred[idx] = self.combiner.combine(ensemble_output, weights)

        return self.ensemble.classes_.take(y_pred)

    def predict_proba(self, X):
        #TODO use a combine_proba to get probabilities (mean rule only)
        X = check_array(X)

        y_probs = np.zeros((X.shape[0], len(self.ensemble.classes_)))

        classes_ = self.ensemble.classes_

        for idx in range(X.shape[0]):
            if self.selector is not None:
                ensemble, weights = self.selector.select(self.ensemble, X[idx,:])
                if ensemble.classes_ != classes_:
                    raise Exception('Selecting subset of classifiers with'
                            'different classes than the original ensemble!'
                            'The selector requires all classifiers to be'
                            'trained with the same classes!')

            else:
                ensemble, weights = self.ensemble, None

            ensemble_output = ensemble.output(X[idx,:].reshape(1,-1))
            # (n_samples, n_classes, n_estimators) to (n_estimators, n_classes)
            ensemble_output = ensemble_output[0,:,:].T
            y_probs[idx,:] = np.mean(ensemble_output, axis=0)

        return y_probs

    def score(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class EnsembleRegressor(object):

    def __init__(self, ensemble=None, combiner='mean'):
        #TODO validate ensemble of regressors
        if ensemble is None or not is_ensemble_of_regressors(ensemble):
            raise ValueError('Estimators in ensemble must be regressors!')

        if isinstance(combiner, str):
            self.combiner = Combiner(rule=combiner)
        elif isinstance(combiner, Combiner):
            self.combiner = combiner
        else:
            raise ValueError('Invalid parameter combiner')

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.ensemble.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        ensemble_output = self.ensemble.output(X)
        return self.combiner.combine(ensemble_output)

    def score(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


