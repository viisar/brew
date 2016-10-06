import numpy as np
from brew.combination import rules


class Combiner(object):

    def __init__(self, rule='majority_vote'):
        self.combination_rule = rule

        if rule == 'majority_vote':
            self.rule = rules.majority_vote_rule

        elif rule == 'max':
            self.rule = rules.max_rule

        elif rule == 'min':
            self.rule = rules.min_rule

        elif rule == 'mean':
            self.rule = rules.mean_rule

        elif rule == 'median':
            self.rule = rules.median_rule

        else:
            raise Exception('invalid argument rule for Combiner class')

    def combine(self, results):

        n_samples = results.shape[0]

        out = np.zeros((n_samples,))

        for i in range(n_samples):
            out[i] = self.rule(results[i, :, :])

        return out
