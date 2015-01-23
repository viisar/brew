import numpy as np

from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier

from brew.base import Ensemble
from .base import DCS

class OLA(DCS):

    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]])

        # intialize variables
        # the the indexes of the KNN of x
        classifiers = ensemble.classifiers
        [idx] = self.knn.kneighbors(x, return_distance=False)
        X, y = self.Xval[idx], self.yval[idx]


        # d[score] = indexes of the classifiers with that score
        d = {}
        scores = [clf.score(X, y) for clf in ensemble.classifiers]
        for i, scr in enumerate(scores):
            d[scr] = d[scr] + [i] if scr in d else [i]
        best_scores = sorted([k for k in d.iterkeys()], reverse=True)

        # if there was a single best classifier, return it
        if len(d[best_scores[0]]) == 1:
            print 'single best'
            i = d[best_scores[0]][0]
            return Ensemble([classifiers[i]])

        print 'not single best'

        options = None
        for j, score in enumerate(best_scores):
            pred = [classifiers[i].predict(x) for i in d[score]]
            pred = np.asarray(pred).flatten()

            bincount = np.bincount(pred)
            if options != None:
                for i in range(len(bincount)):
                    bincount[i] = bincount[i] if i in options else 0

            imx = np.argmax(bincount)
            votes = np.argwhere(bincount == bincount[imx]).flatten()
            count = len(votes)
            if count == 1:
                return Ensemble([classifiers[np.argmax(pred == imx)]])
            elif options == None:
                options = votes

        return Ensemble([classifiers[np.argmax(scores)]])


class OLA2(DCS):
    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]])

        # intialize variables
        # the the indexes of the KNN of x
        classifiers = ensemble.classifiers
        [idx] = self.knn.kneighbors(x, return_distance=False)
        X, y = self.Xval[idx], self.yval[idx]

        scores = np.asarray([clf.score(X, y) for clf in classifiers])

        return Ensemble([classifiers[np.argmax(scores)]])





