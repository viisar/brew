"""
See: Kuncheva L.I., C.J. Whitaker. Measures of diversity in classifier ensembles, Machine Learning , 51 , 2003, 181-207,
"""
import pytest

import numpy as np

from skensemble.metrics.diversity import ClassifiersDiversity

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
    'kw' : 0.2200,                  # Kohavi-Wolpert variance
    'kappa' : 0.0079,               # Interrater agreement
    'e' : 0.7200,                   # Entropy Measure E
    'theta' : 0.0264,               # Difficulty Measure
    'gd' : 0.4365,                  # Generalized Diversity
    'cdf' : 0.4889,                 # Coincidence Failure Diversity
}

atol=0.0001

class TestClassifiersDiversity():

    def test_entropy(self):
        div = ClassifiersDiversity(metric='e')
        div_pred = div.calculate(ensemble_oracle)
        div_true = diversity_values['e']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_kw(self):
        div = ClassifiersDiversity(metric='kw')
        div_pred = div.calculate(ensemble_oracle)
        div_true = diversity_values['kw']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_q(self):
        div = ClassifiersDiversity(metric='q')
        div_pred = div.calculate(ensemble_oracle)
        div_true = diversity_values['q']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_corr(self):
        div = ClassifiersDiversity(metric='rho')
        div_pred = div.calculate(ensemble_oracle)
        div_true = diversity_values['rho']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_disag(self):
        div = ClassifiersDiversity(metric='disagreement')
        div_pred = div.calculate(ensemble_oracle)
        div_true = diversity_values['disagreement']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_df(self):
        div = ClassifiersDiversity(metric='df')
        div_pred = div.calculate(ensemble_oracle)
        div_true = diversity_values['df']
        assert np.isclose(div_pred, div_true, atol=atol) 

    def test_invalid_metric(self):
        pytest.raises(ValueError, ClassifiersDiversity, metric='dsa')

    def test_invalid_oracle_none(self):
        div = ClassifiersDiversity(metric='e')
        pytest.raises(ValueError, div.calculate, None)

    def test_invalid_oracle_shape(self):
        oracle_3d = np.zeros((10,2,1))
        div = ClassifiersDiversity(metric='e')
        pytest.raises(ValueError, div.calculate, oracle_3d)

    def test_invalid_oracle_1_classifier(self):
        div = ClassifiersDiversity(metric='e')
        pytest.raises(ValueError, div.calculate, ensemble_oracle[:,0,np.newaxis])

