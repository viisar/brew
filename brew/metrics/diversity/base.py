import numpy as np

import brew.metrics.diversity.paired as paired
import brew.metrics.diversity.non_paired as non_paired

class Diversity(object):
    
    def __init__(self, metric=''):
        
        if metric == 'entropy':
            self.metric = non_paired.kuncheva_entropy_measure

        elif metric == 'kw':
            self.metric = non_paired.kuncheva_kw

        elif metric == 'q':
            self.metric = paired.kuncheva_q_statistics

        elif metric == 'p':
            self.metric = paired.kuncheva_correlation_coefficient_p

        elif metric == 'disagreement':
            self.metric = paired.kuncheva_disagreement_measure

        elif metric == 'agreement':
            self.metric = paired.kuncheva_agreement_measure

        elif metric == 'df':
            self.metric = paired.kuncheva_double_fault_measure

        else:
            print('invalid metric')


    def calculate(self, ensemble, X, y):
        out = ensemble.output(X, mode='labels')
        oracle = np.equal(out, y[:,np.newaxis])

        D = self.metric(oracle)

        return D

