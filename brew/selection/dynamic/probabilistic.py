import numpy as np

from brew.base import Ensemble
from .base import DCS

class Priori(DCS):

    def __init__(self, threshold=0.05):
        self.threshold = threshold
        
    def probabilities(self, clf, nn_X, nn_y, distances):
        proba = clf.predict_proba(nn_X)
        proba = np.hstack((proba), np.zeros(len(proba),1))

        d = dict(list(enumerate(clf.classes_)))
        col_idx = np.zeros(nn_y.size)
        for i in range(nn_y.size):
            col_idx[i] = d[nn_y[i]] if nn_y[i] in d else proba.shape[1] - 1

        probabilities = proba[np.arange(col_idx.size), col_idx]
        delta = 1./(distances + 10e-8)
        
        p_correct = np.sum(probabilities * delta) / np.sum(delta)
        return p_correct

    def select(self, ensemble, x):
        selected_classifier = None

        nn_X, nn_y, dists = self.get_neighbors(x, 
                return_distance=True)
        
        idx_selected, prob_selected = [], []
        
        for idx, clf in enumerate(ensemble.classifiers):
            prob = self.probability(clf, nn_X, nn_y, dists)
            if prob > 0.5:
                idx_selected = idx_selected + [idx]
                prob_selected = prob_selected + [prob]
        
        p_correct_m = max(prob_selected)
        m = np.argmax(prob_selected)

        selected = True
        diffs = []
        for j, p_correct_j in enumerate(prob_selected):
            d = p_correct_m - p_correct_j
            diffs.append(d)
            if j != m and d < self.threshold:
                selected = False

        if selected:
            selected_classifier = ensemble.classifiers[idx_selected[m]]
        else:
            i = np.random.choice(idx_selected[np.array(diffs) < self.threshold])
            selected_classifier = ensemble.classifiers[i]
        
        return Ensemble([selected_classifier]), None



class Posteriori(DCS):

    def select(self, ensemble, x):
        pass

