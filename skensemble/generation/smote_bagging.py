from __future__ import division

import numpy as np

import sklearn
from sklearn.neighbors import KNeighborsClassifier

from brew.base import Ensemble
from brew.combination.combiner import Combiner
from brew.preprocessing.smote import smote

from .base import PoolGenerator


class SmoteBagging(PoolGenerator):

    def __init__(self, base_classifier=None,
                 n_classifiers=100,
                 combination_rule='majority_vote', k=5):

        # self.b = b
        self.k = k
        self.n_classifiers = n_classifiers
        self.base_classifier = base_classifier

        self.ensemble = None
        self.combiner = Combiner(rule=combination_rule)

    def smote_bootstrap_sample(self, X, y, b, k):

        classes = np.unique(y)
        count = np.bincount(y)  # number of instances of each class

        majority_class = count.argmax()  # majority clas
        majority_count = count.max()  # majority class

        data = np.empty((0, X.shape[1]))
        target = np.empty((0,))

        for i in classes:

            class_data = X[(y == i), :]

            if i == majority_class:  # majority class
                # regular bootstrap (i.e. 100% sampling rate)
                idx = np.random.choice(majority_count, (majority_count,))
                data = np.concatenate((data, class_data[idx, :]))
                target = np.concatenate(
                    (target, i * np.ones((majority_count,))))
                # print('original class data = {}'.format(class_data.shape))
                # print('sampled class data = {}'.format(class_data[idx,:].shape))  # noqa
                # print()

            else:  # minority classes
                # bootstrap the class data with defined sampling rate
                sample_rate = (majority_count /
                               class_data.shape[0]) * (b / 100)
                idx = np.random.choice(
                    class_data.shape[0], (int(sample_rate * class_data.shape[0]),))  # noqa
                sampled_class_data = class_data[idx, :]

                # print('original class data = {}'.format(class_data.shape))
                # print('majority_count = {}'.format(majority_count))
                # print('class data = {}'.format(class_data.shape))
                # print('b = {}'.format(b))
                # print('sample rate = {}'.format(sample_rate))
                # print('sampled class data = {}'.format(sampled_class_data.shape)) # noqa

                # run smote on bootstrapped data to obtain synthetic samples
                # ceil to make sure N_smote is a multiple of 100, and the small
                # value to avoid a zero
                N_smote = int(np.ceil(
                    (majority_count / sampled_class_data.shape[0]) * (1 - b / 100 + 10e-8)) * 100)  # noqa
                # print(N_smote)

                # print('----------')
                # print('smote parameters:')
                # print('T : {}'.format(sampled_class_data.shape))
                # print('N : {}'.format(N_smote))
                synthetic = smote(sampled_class_data, N=N_smote, k=self.k)
                # print('synthetic data = {})'.format(synthetic.shape))
                # print(synthetic)

                # add synthetic samples to sampled class data
                n_missing = majority_count - sampled_class_data.shape[0]
                idx = np.random.choice(synthetic.shape[0], (n_missing,))
                new_class_data = np.concatenate(
                    (sampled_class_data, synthetic[idx, :]))
                # print('new class data = {})'.format(new_class_data.shape))
                # print()
                data = np.concatenate((data, new_class_data))
                target = np.concatenate(
                    (target, i * np.ones((new_class_data.shape[0],))))

        return data, target

    def fit(self, X, y):

        self.ensemble = Ensemble()

        # this parameter should change between [10, 100] with
        # increments of 10, for every classifier in the ensemble
        b = 10

        for i in range(self.n_classifiers):
            # print()
            # print('classifier : {}'.format(i))
            # print('------------------------')
            # print('b = {}'.format(b))
            data, target = self.smote_bootstrap_sample(
                X, y, b=float(b), k=self.k)
            # print('data = {}'.format(data.shape))
            # print()

            classifier = sklearn.base.clone(self.base_classifier)
            classifier.fit(data, target)

            self.ensemble.add(classifier)

            if b >= 100:
                b = 10
            else:
                b += 10

        return

    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)


class SmoteBaggingNew(SmoteBagging):

    def fit(self, X, y):

        self.ensemble = Ensemble()

        # this parameter should change between [10, 100] with
        # increments of 10, for every classifier in the ensemble
        b = 10

        for i in range(self.n_classifiers):
            # print()
            # print('classifier : {}'.format(i))
            # print('------------------------')
            # print('b = {}'.format(b))
            data, target = self.smote_bootstrap_sample(
                X, y, b=float(b), k=self.k)
            # print('data = {}'.format(data.shape))
            # print()

            classifier = sklearn.base.clone(self.base_classifier)
            classifier.fit(data, target)

            self.ensemble.add(classifier)

            if b >= 100:
                b = 10
            else:
                b += 10

        return

    def smote_bootstrap_sample(self, X, y, b, k):

        count = np.bincount(y)  # number of instances of each class

        majority_class = count.argmax()  # majority class
        majority_count = count.max()  # majority class

        data = np.empty((0, X.shape[1]))
        target = np.empty((0,))

        class_data = X[(y == majority_class), :]
        idx = np.random.choice(majority_count, (majority_count,))
        data = np.concatenate((data, class_data[idx, :]))
        target = np.concatenate(
            (target, majority_class * np.ones((majority_count,))))

        minority_class = count.argmin()
        minority_count = count.min()

        # print majority_count
        N_syn = int((majority_count) * (b / 100))
        # print N_syn
        N_res = majority_count - N_syn
        # print N_res
        N_syn, N_res = N_res, N_syn

        class_data = X[(y == minority_class), :]
        idx = np.random.choice(class_data.shape[0], (N_res,))
        sampled_min_data = class_data[idx, :]
        # print sampled_min_data.shape
        if N_syn > 0:
            N_smote = np.ceil(N_syn / minority_count) * 100
            N_smote = 100 if N_smote < 100 else int(N_smote - N_smote % 100)
            synthetic = smote(X[y == minority_class], N=int(N_smote), k=self.k)

            idx = np.random.choice(synthetic.shape[0], (N_syn,))
            new_class_data = np.concatenate(
                (sampled_min_data, synthetic[idx, :]))
            data = np.concatenate((data, new_class_data))
            target = np.concatenate(
                (target, minority_class * np.ones((new_class_data.shape[0],))))
        else:
            data = np.concatenate((data, sampled_min_data))
            target = np.concatenate(
                (target, minority_class * np.ones((sampled_min_data.shape[0],))))  # noqa

        return data, target


if __name__ == '__main__':
    # class 0
    X0 = np.random.random((100, 2))
    y0 = 0 * np.ones((100,), dtype='int64')

    # class 1
    X1 = np.random.random((60, 2))
    y1 = 1 * np.ones((60,), dtype='int64')

    # class 2
    X2 = np.random.random((35, 2))
    y2 = 2 * np.ones((35,), dtype='int64')

    # class 3
    X3 = np.random.random((5, 2))
    y3 = 3 * np.ones((5,), dtype='int64')

    print('DATASET before:')
    print('class 0 : {}'.format(X0.shape))
    print('class 1 : {}'.format(X1.shape))
    print('class 2 : {}'.format(X2.shape))
    print('class 3 : {}'.format(X3.shape))
    print()

    X = np.concatenate((X0, X1, X2, X3))
    y = np.concatenate((y0, y1, y2, y3))

    knn = KNeighborsClassifier
    pool = SmoteBagging(base_classifier=knn, n_classifiers=5, k=3)
    pool.fit(X, y)

    print(np.sum(pool.predict(X) == y) / y.size)
