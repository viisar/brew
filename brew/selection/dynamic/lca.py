import numpy as np

from brew.base import Ensemble
from .base import DCS

class LCA(DCS):
    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]])

        # obtain the K nearest neighbors in the validation set
        [idx] = self.knn.kneighbors(x, return_distance=False)
        neighbors_X = self.Xval[idx] # k neighbors
        neighbors_y = self.yval[idx] # k neighbors target

        # pool_output (sample, classifier_output)
        pool_output = np.zeros((neighbors_X.shape[0], len(ensemble)))
        for i, clf in enumerate(ensemble.classifiers):
            pool_output[:,i] = clf.predict(neighbors_X)

        x_outputs = [ensemble.classifiers[j].predict(x) for j in len(ensemble)]
        x_outputs = np.asarray(x_outputs).flatten()

        d = {}
        scores = np.zeros(len(ensemble))
        for j in range(pool_output.shape[1]):
            # get correctly classified samples
            mask = pool_output[:,j] == neighbors_y
            # get 
            mask = (pool_output[:,j] == x_outputs[j]) * mask
            scores[j] = sum(mask)
            d[scores[j]] = d[scores[j]] + 1 if scores[j] in d else 1

        best_scores = sorted([k for k in d.iterkeys()], reverse=True)
        
        options = None
        for j, score in enumerate(best_scores):
            pred = [x_outputs[i] for i in d[score]]
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

           


def LCA2(DCS):
    def select(self, ensemble, x):
        if ensemble.in_agreement(x):
            return Ensemble([ensemble.classifiers[0]])

        # obtain the K nearest neighbors in the validation set
        [idx] = self.knn.kneighbors(x, return_distance=False)
        neighbors_X = self.Xval[idx] # k neighbors
        neighbors_y = self.yval[idx] # k neighbors target

        # pool_output (sample, classifier_output)
        pool_output = np.zeros((neighbors_X.shape[0], len(ensemble)))
        for i, clf in enumerate(ensemble.classifiers):
            pool_output[:,i] = clf.predict(neighbors_X)

        x_outputs = [ensemble.classifiers[j].predict(x) for j in len(ensemble)]
        x_outputs = np.asarray(x_outputs).flatten()

        scores = np.zeros(len(ensemble))
        for j in range(pool_output.shape[1]):
            # get correctly classified samples
            mask = pool_output[:,j] == neighbors_y
            # get 
            mask = (pool_output[:,j] == x_outputs[j]) * mask
            scores[j] = sum(mask)
            
        return Ensemble([classifiers[np.argmax(scores)]])
        

        

        









