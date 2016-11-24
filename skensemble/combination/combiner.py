import numpy as np
from skensemble.combination import rules
from skensemble.ensemble import output2votes


class Combiner(object):

    def __init__(self, rule='majority_vote'):
        self.combination_rule = rule
        self.__rule = None

        if rule == 'majority_vote':
            self.__rule = rules.majority_vote_rule

        elif rule == 'max':
            self.__rule = rules.max_rule

        elif rule == 'min':
            self.__rule = rules.min_rule

        elif rule == 'mean':
            self.__rule = rules.mean_rule

        elif rule == 'median':
            self.__rule = rules.median_rule

        else:
            raise Exception('invalid argument rule for Combiner class')

    def combine(self, ensemble_output):
        # majority vote does not make sense with probabilities output
        if self.combination_rule == 'majority_vote':
            ensemble_output = output2votes(ensemble_output)

        n_samples = ensemble_output.shape[0]
        out = np.zeros((n_samples,))

        for i in range(n_samples):
            out[i] = self.__rule(results[i, :, :])

        return out
