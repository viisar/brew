"""
See: Kuncheva L.I., C.J. Whitaker. Measures of diversity in classifier ensembles, Machine Learning , 51 , 2003, 181-207,
"""

import numpy as np

from brew.metrics.diversity import paired
from brew.metrics.diversity import non_paired

# 'oracle' style
# rows are instances, cols are classifiers, elements are boolean
# values indicating if the classifier got the answer right

ensemble_output = np.array([
     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
     [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
     [1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
     [1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
     [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
     [1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
     [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
     [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
     [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
     [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
     [1, 1, 1, 1, 0, 0, 1, 0, 0, 1] ])

example = {}
# --------- paired -----------
example['Q']       = 0.0231   # Q-statistic
example['rho']     = 0.0156   # Correlation coefficient
example['disag']   = 0.4889   # Disagreement

# Agreement 
example['ag']      = 1.0 / example['disag']

example['DF']      = 0.3156   # Double Fault
# ------- non-paired ---------
example['KW']      = 0.2200   # Kohavi-Wolpert variance
example['kappa']   = 0.0079   # Interrater agreement
example['entropy'] = 0.7200   # Entropy Measure E
example['theta']   = 0.0264   # Difficulty Measure
example['GD']      = 0.4365   # Generalized Diversity
example['CFD']     = 0.4889   # Coincidence Failure Diversity

atol=0.0001


class TestNonPaired():

    def test_entropy(self):
        assert np.isclose(non_paired.kuncheva_entropy_measure(ensemble_output), example['entropy'], atol=atol) 

    def test_kw(self):
        assert np.isclose(non_paired.kuncheva_kw(ensemble_output), example['KW'], atol=atol)


class TestPaired():

    def test_q(self):
        assert np.isclose(paired.kuncheva_q_statistics(ensemble_output), example['Q'], atol=atol)

    def test_corr(self):
        assert np.isclose(paired.kuncheva_correlation_coefficient_p(ensemble_output), example['rho'], atol=atol)

    def test_disag(self):
        assert np.isclose(paired.kuncheva_disagreement_measure(ensemble_output), example['disag'], atol=atol)

    def test_ag(self):
        assert np.isclose(paired.kuncheva_agreement_measure(ensemble_output), example['ag'], atol=atol)

    def test_df(self):
        assert np.isclose(paired.kuncheva_double_fault_measure(ensemble_output), example['DF'], atol=atol)





