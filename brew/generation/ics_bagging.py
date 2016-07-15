from __future__ import division

import numpy as np

import sklearn
from sklearn.metrics import recall_score

from brew.base import Ensemble
from brew.combination.combiner import Combiner
from brew.preprocessing.smote import smote
from brew.metrics.diversity.base import Diversity
import brew.metrics.evaluation as evaluation

from .base import PoolGenerator


class ICSBagging(PoolGenerator):

    def __init__(self,
                 K=10,
                 alpha=0.75,
                 base_classifier=None,
                 n_classifiers=100,
                 combination_rule='majority_vote',
                 diversity_metric='e',
                 positive_label=1):

        self.K = K
        self.alpha = alpha

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.combination_rule = combination_rule
        self.positive_label = positive_label

        self.classifiers = None
        self.ensemble = None
        self.combiner = Combiner(rule=combination_rule)

        self.diversity_metric = diversity_metric
        self.diversity = Diversity(metric=diversity_metric)

        self.validation_X = None
        self.validation_y = None

    def set_validation(self, X, y):
        self.validation_X = X
        self.validation_y = y

    def fitness(self, classifier):
        '''
        # TODO normalize diversity metric.
        '''
        self.ensemble.add(classifier)
        out = self.ensemble.output(self.validation_X)
        y_pred = self.combiner.combine(out)
        y_true = self.validation_y

        auc = evaluation.auc_score(y_true, y_pred)
        div = self.diversity.calculate(self.ensemble,
                                       self.validation_X, self.validation_y)

        # diversity = entropy_measure_e(self.ensemble,
        #        self.validation_X, self.validation_y)

        self.ensemble.classifiers.pop()
        return self.alpha * auc + (1.0 - self.alpha) * div

    def _calc_pos_prob(self):
        y_pred = self.combiner.combine(self.ensemble.output(self.validation_X))
        mask = self.positive_label == self.validation_y
        pos_acc = float(sum(y_pred[mask] == self.validation_y[
                        mask])) / len(self.validation_y[mask])
        neg_acc = float(sum(y_pred[~mask] == self.validation_y[
                        ~mask])) / len(self.validation_y[~mask])
        return 1.0 - (pos_acc / (pos_acc + neg_acc))

    def bootstrap_classifiers(self, X, y, K, pos_prob):
        mask = self.positive_label == y
        negative_label = y[~mask][0]

        clfs = []
        sets_cX, sets_cy = [], []
        for i in range(K):
            cX, cy = [], []
            for j in range(X.shape[0]):
                if np.random.random() < pos_prob:
                    idx = np.random.random_integers(0, len(X[mask]) - 1)
                    cX = cX + [X[mask][idx]]
                    cy = cy + [self.positive_label]
                else:
                    idx = np.random.random_integers(0, len(X[~mask]) - 1)
                    cX = cX + [X[~mask][idx]]
                    cy = cy + [negative_label]
            if self.positive_label not in cy:
                idx_1 = np.random.random_integers(0, len(cX) - 1)
                idx_2 = np.random.random_integers(0, len(X[mask]) - 1)
                cX[idx_1] = X[mask][idx_2]
                cy[idx_1] = self.positive_label
            elif negative_label not in cy:
                idx_1 = np.random.random_integers(0, len(cX) - 1)
                idx_2 = np.random.random_integers(0, len(X[~mask]) - 1)
                cX[idx_1] = X[~mask][idx_2]
                cy[idx_1] = negative_label
            # print len(cX), len(cy), X.shape[0], len(X), np.bincount(cy)

            sets_cX, sets_cy = sets_cX + [cX], sets_cy + [cy]
            clf = sklearn.base.clone(self.base_classifier)
            clfs = clfs + [clf.fit(cX, cy)]

        return clfs

    def fit(self, X, y):
        # if self.validation_X == None and self.validation_y == None:
        self.validation_X = X
        self.validation_y = y

        self.classes_ = set(y)
        self.ensemble = Ensemble()

        clfs = self.bootstrap_classifiers(X, y, self.K, 0.5)
        self.ensemble.add(np.random.choice(clfs))

        for _ in range(1, self.n_classifiers):
            clfs = self.bootstrap_classifiers(
                X, y, self.K, self._calc_pos_prob())
            self.ensemble.add(max(clfs, key=lambda clf: self.fitness(clf)))

        self.validation_X = None
        self.validation_y = None

        return self

    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)


class ICSBaggingNew(PoolGenerator):

    def __init__(self,
                 K=10,
                 alpha=0.75,
                 base_classifier=None,
                 n_classifiers=100,
                 combination_rule='majority_vote',
                 diversity_metric='e',
                 positive_label=1):

        self.K = K
        self.alpha = alpha

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.positive_label = positive_label

        self.ensemble = None
        self.combiner = Combiner(rule=combination_rule)

        self.diversity = Diversity(metric=diversity_metric)

        self.validation_X = None
        self.validation_y = None

    def set_validation(self, X, y):
        self.validation_X = X
        self.validation_y = y

    def fitness(self, classifier):
        '''
        # TODO normalize diversity metric.
        '''
        self.ensemble.add(classifier)

        y_pred = self.predict(self.validation_X)
        y_true = self.validation_y

        auc = evaluation.auc_score(y_true, y_pred)
        div = self.diversity.calculate(
            self.ensemble, self.validation_X, y_true)

        self.ensemble.classifiers.pop()  # create interface for this later

        return self.alpha * auc + (1.0 - self.alpha) * div

    def _calc_pos_prob(self):
        y_pred = self.predict(self.validation_X)
        y_true = self.validation_y

        # obtaining recall scores for each label (assuming the labels are
        # binary)
        pos_acc = recall_score(
            y_true, y_pred, average='binary', pos_label=self.positive_label)
        neg_acc = recall_score(y_true,
                               y_pred,
                               average='binary',
                               pos_label=int(not self.positive_label))

        return neg_acc / (pos_acc + neg_acc)

    def bootstrap_classifiers(self, X, y, K, pos_prob):
        pos_idx = (y == self.positive_label)
        neg_idx = (y == int(not self.positive_label))

        X_pos, _ = X[pos_idx, :], y[pos_idx]  # positive examples
        X_neg, _ = X[neg_idx, :], y[neg_idx]  # negative examples

        classifiers = []
        for i in range(K):
            X_new = np.zeros(X.shape)
            y_new = np.zeros(y.shape)

            for j in range(X.shape[0]):

                if pos_prob > np.random.random():
                    # add a randomly chosen positive example
                    idx = np.random.randint(X_pos.shape[0])
                    X_new[j, :] = X_pos[idx, :]
                    y_new[j] = self.positive_label

                else:
                    # add a randomly chosen negative example
                    idx = np.random.randint(X_neg.shape[0])
                    X_new[j, :] = X_neg[idx, :]
                    y_new[j] = int(not self.positive_label)

            # if no positive example is present, make sure you insert at least
            # one
            if not np.any(y_new == self.positive_label):
                # chosen spot for replacement on new array
                idx_new = np.random.randint(X_new.shape[0])
                # chosen positive example index
                idx_pos = np.random.randint(X_pos.shape[0])

                X_new[idx_new, :] = X_pos[idx_pos, :]
                y_new[idx_new] = self.positive_label

            # if no negative example is present, make sure you insert at least
            # one
            elif not np.any(y_new == int(not self.positive_label)):
                # chosen spot for replacement on new array
                idx_new = np.random.randint(X_new.shape[0])
                # chosen positive example index
                idx_neg = np.random.randint(X_neg.shape[0])

                X_new[idx_new, :] = X_neg[idx_neg, :]
                y_new[idx_new] = int(not self.positive_label)

            # train classifier with the bootstrapped data
            clf = sklearn.base.clone(self.base_classifier)
            clf.fit(X_new, y_new)

            classifiers.append(clf)

        return classifiers

    def fit(self, X, y):
        # if self.validation_X == None and self.validation_y == None:
        self.validation_X = X
        self.validation_y = y

        self.classes_ = set(y)
        self.ensemble = Ensemble()

        clfs = self.bootstrap_classifiers(X, y, self.K, 0.5)
        self.ensemble.add(np.random.choice(clfs))

        for i in range(1, self.n_classifiers):
            clfs = self.bootstrap_classifiers(
                X, y, self.K, self._calc_pos_prob())
            self.ensemble.add(max(clfs, key=lambda clf: self.fitness(clf)))

        self.validation_X = None
        self.validation_y = None

        return self

    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)


class SmoteICSBagging(ICSBagging):

    def __init__(self,
                 K=10,
                 alpha=0.75,
                 base_classifier=None,
                 n_classifiers=100,
                 combination_rule='majority_vote',
                 diversity_metric='e',
                 positive_label=1,
                 smote_k=5):
        self.smote_k = smote_k
        super(SmoteICSBagging, self).__init__(K=K,
                                              alpha=alpha,
                                              base_classifier=base_classifier,
                                              n_classifiers=n_classifiers,
                                              combination_rule=combination_rule,  # noqa
                                              diversity_metric=diversity_metric,  # noqa
                                              positive_label=positive_label)

    def bootstrap_classifiers(self, X, y, K, pos_prob):

        clfs = []

        for i in range(K):
            mask = (self.positive_label == y)
            negative_label = y[~mask][0]

            majority_size = np.sum(~mask)
            minority_size = len(mask) - majority_size

            # apply smote
            N_smote = int(np.ceil(majority_size / minority_size) * 100)

            # print 'classifier: {}'.format(i)
            # print '     maj size = {}'.format(majority_size)
            # print '     min size = {}'.format(minority_size)
            # print '     SMOTE:'
            # print '         N_smote: {}'.format(N_smote)
            # print '         T : {}'.format(X[mask].shape)

            X_syn = smote(X[mask], N=N_smote, k=self.smote_k)
            # print '         out : {}'.format(X_syn.shape)
            y_syn = self.positive_label * np.ones((X_syn.shape[0],))

            # use enough synthetic data to perfectly balance the binary problem
            n_missing = majority_size - minority_size
            # print n_missing
            idx = np.random.choice(X_syn.shape[0], n_missing)

            # add synthetic data to original data
            X_new = np.concatenate((X, X_syn[idx, ]))
            y_new = np.concatenate((y, y_syn[idx, ]))

            # use new mask
            mask = (self.positive_label == y_new)

            # balance the classes

            cX, cy = [], []
            for j in range(X_new.shape[0]):
                if np.random.random() < pos_prob:
                    idx = np.random.random_integers(0, len(X_new[mask]) - 1)
                    cX = cX + [X_new[mask][idx]]
                    cy = cy + [self.positive_label]
                else:
                    idx = np.random.random_integers(0, len(X_new[~mask]) - 1)
                    cX = cX + [X_new[~mask][idx]]
                    cy = cy + [negative_label]
            if self.positive_label not in cy:
                idx_1 = np.random.random_integers(0, len(cX) - 1)
                idx_2 = np.random.random_integers(0, len(X_new[mask]) - 1)
                cX[idx_1] = X_new[mask][idx_2]
                cy[idx_1] = self.positive_label
            elif negative_label not in cy:
                idx_1 = np.random.random_integers(0, len(cX) - 1)
                idx_2 = np.random.random_integers(0, len(X_new[~mask]) - 1)
                cX[idx_1] = X_new[~mask][idx_2]
                cy[idx_1] = negative_label

            clf = sklearn.base.clone(self.base_classifier)
            clfs = clfs + [clf.fit(cX, cy)]

        return clfs
