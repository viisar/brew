"""
See: Kuncheva L.I., C.J. Whitaker. Measures of diversity in classifier ensembles, Machine Learning , 51 , 2003, 181-207,
"""
import pytest

import numpy as np

from skensemble.metrics.diversity import non_paired

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
    'kw' : 0.2200,                  # Kohavi-Wolpert variance
    'e' : 0.7200,                   # Entropy Measure E
}

atol=0.0001

class TestNonPaired():

    def test_entropy(self):
        div_pred = non_paired.entropy_e(ensemble_oracle)
        div_true = diversity_values['e']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_kw(self):
        div_pred = non_paired.kohavi_wolpert_variance(ensemble_oracle)
        div_true = diversity_values['kw']
        assert np.isclose(div_pred, div_true, atol=atol) 

