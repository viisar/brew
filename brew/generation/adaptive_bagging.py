import numpy as np

import sklearn
from sklearn.lda import LDA
from sklearn.decomposition import PCA

from brew.base import Ensemble
from brew.combination.rules import majority_vote_rule
from brew.combination.combiner import Combiner
from brew.generation import Bagging

from brew.metrics.evaluation import auc_score
from brew.metrics.diversity.paired import paired_metric_ensemble

from .base import PoolGenerator

class AdaptiveBagging(PoolGenerator):


    def __init__(self, K=10, alpha=0.75, base_classifier=None, n_classifiers=100,
            combination_rule='majority_vote', max_samples=1.0, positive_label = 1):

        self.K = K
        self.alpha = alpha

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.combination_rule = combination_rule
        self.positive_label = positive_label
        self.bagging = Bagging(base_classifier=base_classifier, n_classifiers=K)

        self.classifiers = None
        self.ensemble = None
        self.combiner = Combiner(rule=combination_rule)

        self.validation_X = None
        self.validation_y = None


    def set_validation(self, X, y):
        self.validation_X = X
        self.validation_y = y


    def fitness(self, classifier):
        #TODO check different diversities and normalize
        self.ensemble.add(classifier)
        out = self.ensemble.output(self.validation_X)
        y_pred = self.combiner.combine(out)
        y_true = self.validation_y
        auc = auc_score(y_true, y_pred)

        diversity = paired_metric_ensemble(ensemble, 
                self.validation_X, self.validation_y)
         
        self.ensemble.classifiers.pop()
        return self.alpha * auc + (1.0 - self.alpha) * diversity


    def _calc_pos_prob(self):
        y_pred = self.combiner.combine(self.ensemble.output(self.validation_X))
        mask = self.positive_label == y
        pos_acc = float(sum(y_pred[mask] == y[mask]))/len(y[mask])
        neg_acc = float(sum(y_pred[~mask] == y[~mask]))/len(y[~mask])
        return 1.0 - (pos_acc / (pos_acc + neg_acc))


    def bootstrap_classifiers(self, X, y, K, pos_prob):
        mask = self.positive_label == y
        negative_label = y[~mask][0]

        clfs = []
        for i in range(K):
            for j in range(X.shape[0]):
                cX, cy = [], []
                if np.random.random() < pos_prob:
                    idx = np.random.random_integers(0, len(X[mask]) - 1)
                    cX = cX + [X[mask][idx]]
                    cy = cy + [self.positive_label]
                else:
                    idx = np.random.random_integers(0, len(X[~mask]) - 1)
                    cX = cX + [X[~mask][idx]]
                    cy = cy + [negative_label]
            if not self.positive_label in cy:
                idx_1 = np.random.random_integers(0, len(cX) - 1)
                idx_2 = np.random.random_integers(0, len(X[mask])- 1)
                cX[idx_1] = X[mask][idx_2]
                cy[idx_1] = negative_label
            elif not negative_label in cy:
                idx_1 = np.random.random_integers(0, len(cX) - 1)
                idx_2 = np.random.random_integers(0, len(X[~mask])- 1)
                cX[idx_1] = X[~mask][idx_2]
                cy[idx_1] = self.positive_label

            clf = sklearn.base.clone(self.base_classifier)
            clfs = clfs + [clf.fit(cX, cy)]

        return clfs


    def fit(self, X, y):
        if self.validation_X == None and self.validation_y == None:
            self.validation_X = X
            self.validation_y = y

        self.classes_ = set(y)
        self.ensemble = Ensemble()

        clfs = self.bootstrap_classifiers(X, y, self.K, 0.5)
        self.ensemble.add(np.random.choice(clfs))

        for i in range(1, self.n_classifiers):
            clfs = self.bootstrap_classifiers(X, y, self.K, self._calc_pos_prob())
            self.ensemble.add(max(clfs, key=lambda clf: self.fitness(clf)))
        
        return self


    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)



