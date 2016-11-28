"""
See: Kuncheva L.I., C.J. Whitaker. Measures of diversity in classifier ensembles, Machine Learning , 51 , 2003, 181-207,
"""
import pytest

import numpy as np

from skensemble.metrics.diversity import paired

ensemble_oracle = np.array([
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

diversity_values = {
    'q' : 0.0231,                   # Q-statistic
    'rho' : 0.0156,                 # Correlation coefficient
    'disagreement' : 0.4889,        # Disagreement
    'df' : 0.3156,                  # Double Fault
}

atol=0.0001

class TestPaired():

    def test_q(self):
        div_pred = paired.q_statistics(ensemble_oracle)
        div_true = diversity_values['q']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_corr(self):
        div_pred = paired.correlation_coefficient_rho(ensemble_oracle)
        div_true = diversity_values['rho']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_disag(self):
        div_pred = paired.disagreement(ensemble_oracle)
        div_true = diversity_values['disagreement']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_df(self):
        div_pred = paired.double_fault(ensemble_oracle)
        div_true = diversity_values['df']
        assert np.isclose(div_pred, div_true, atol=atol) 

