import numpy as np
from skensemble.combination import classification_rules
from skensemble.combination import regression_rules

from skensemble.ensemble import output2votes


CLASSIFICATION_COMBINATION_RULES = [
    'majority_vote',
    'max',
    'min',
    'mean',
    'median'
]

REGRESSION_COMBINATION_RULES = [
    'max',
    'min',
    'mean',
    'median'
]

RULES_ALLOW_WEIGHTS = [
    'majority_vote',
    'mean',
    'median'
]


class Combiner(object):
    def __init__(self, rule='majority_vote', weights=None):
        self.__validate_weights(None, rule, weights)

        self.combination_rule = rule
        self.weights = weights
        self.__rule = None

        if rule not in CLASSIFICATION_COMBINATION_RULES and
           rule not in REGRESSION_COMBINATION_RULES:
            raise Exception('invalid argument rule for Combiner class')


    def __validate_weights(ensemble_output, rule, weights):
        if weights is not None:
            if rule not in RULES_ALLOW_WEIGHTS:
                raise ValueError('The requested combination rule does not accept weights')
            else:
                if ensemble_output is not None and weights.size != ensemble_output.shape[-1]:
                    raise ValueError('Weight vector length must be equal to the number of estimators')


    def __validate_setup(ensemble_output, rule, weights):
        if weights:
            self.__validate_weight(ensemble_output, rule, weights)
        else:
            weights = self.weights
            self.__validate_weight(ensemble_output, rule, weights)
       
        # ensemble of classifiers
        if ensemble_output.ndim == 3:
            if rule not in CLASSIFICATION_COMBINATION_RULES:
                raise ValueError('Chosen combination rule does not work with classification.')
            
        # ensemble of regressors
        elif ensemble_output.ndim == 2:
            if rule not in REGRESSION_COMBINATION_RULES:
                raise ValueError('Chosen combination rule does not work with regression.')
        
        else:
            raise ValueError('Ensemble output shape is invalid, expecting 2d or 3d array')


    def combine(self, ensemble_output, weights=None):
        self.__validate_setup(ensemble_output, self.combination_rule, weights)

        rule_func = None

        # ensemble of classifiers
        if ensemble_output.ndim == 3:
            rule_func = classification_rules.RULE_FUNCTIONS.get(self.combination_rule)
        
        # ensemble of regressors
        elif ensemble_output.ndim == 2:
            rule_func = regression_rules.RULE_FUNCTIONS.get(self.combination_rule)
       
        else:
            raise ValueError('Ensemble output shape is invalid, expecting 2d or 3d array')

        # majority vote does not make sense with probabilities output
        if self.combination_rule == 'majority_vote':
            ensemble_output = output2votes(ensemble_output)

        n_samples = ensemble_output.shape[0]

        out = np.zeros((n_samples,))
        for i in range(n_samples):
            sample = results[i, ...] * weights
            out[i] = rule_func(sample)

        return out
