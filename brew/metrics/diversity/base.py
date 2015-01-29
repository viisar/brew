import numpy as np

import brew.metrics.diversity.paired as paired
import brew.metrics.diversity.non_paired as non_paired


class Diversity(object):
    
    def __init__(self, metric=''):
        
        if metric == 'entropy':
            self.metric = non_paired.kuncheva_entropy_measure

        elif metric == 'kw':
            self.metric = non_paired.kuncheva_kw

    def calculate(self, ensemble, X, y):
        out = ensemble.output(X, mode='labels')
        oracle = np.equal(out, y[:,np.newaxis])

        D = self.metric(oracle)

        return D


