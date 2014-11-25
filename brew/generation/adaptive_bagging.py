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
            combination_rule='majority_vote', max_samples=1.0):

        self.K = K
        self.alpha = alpha

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.combination_rule = combination_rule
        self.bagging = Bagging(base_classifier=base_classifier, n_classifiers=K)

        self.classifiers = None
        self.ensemble = None
        self.combiner = Combiner(rule=combination_rule)
        

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
         
        self.ensemble.classifiers = self.ensemble.classifiers[:-1]
        return self.alpha * auc + (1.0 - self.alpha) * diversity


    def fit(self, X, y):
        if self.validation_X == None and self.validation_y == None:
            self.validation_X = X
            self.validation_y = y
        
        classifiers = []     
    
        self.bagging.fit(X, y)
        classifiers.append(np.random.choice(clfs))

        for i in range(self.n_classifiers):
            #TODO use weights for classes here
            self.bagging.fit(X, y)
            clfs = self.bagging.ensemble.classifiers
            if i == 0:
                classifiers.append(np.random.choice(clfs))
            else:
                mx = -1
                for idx, clf in enumerate(clfs):
                    ft = self.fitness(clf)
                    if ft > idx:
                        mx = idx
                classifiers.append(clfs[mx])
        
        self.ensemble = Ensemble()
        self.ensemble.add_classifiers(classifiers)
        return self

    def predict(self, X):
        out = self.ensemble.output(X)
        return self.combiner.combine(out)
            
