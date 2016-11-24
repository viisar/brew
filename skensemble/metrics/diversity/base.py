import numpy as np

from brew.metrics.diversity import paired
from brew.metrics.diversity import non_paired

CLASSIFICATION_DIVERSITY_FUNCTIONS = {
    'e' : non_paired.entropy_e,
    'kw' : non_paired.kohavi_wolpert_variance,
    'q' : paired.q_statistics,
    'rho' : paired.correlation_coefficient_rho,
    'disagreement' : paired.disagreement,
    'agreement' : paired.agreement,
    'df' : paired.double_fault
}

class ClassifiersDiversity(object):
    """Diversity of Classifiers Ensemble.

    Calculates the diversity of ensemble of classifiers.

    Attributes
    ----------
    `metric` : function, receives the oracle output and returns float
        Function used to calculate the metric.

    Parameters
    ----------
    metric : str or function, (default = 'e')
        Metric used to compute the ensemble diversity:

        - 'e' (Entropy Measure e) will use :meth:`entropy_e`
        - 'kw' (Kohavi Wolpert Variance) will use :meth:`kohavi_wolpert_variance`
        - 'q' (Q Statistics) will use :meth:`q_statistics`
        - 'rho' (Correlation Coefficient Rho) will use :meth:`correlation_coefficient_rho`
        - 'disagreement' (Disagreement Measure) will use :meth:`disagreement`
        - 'agreement' (Agreement Measure) will use :meth:`agreement`
        - 'df' (Double Fault Measure) will use :meth:`double_fault`

    Examples
    --------
    >>> from brew.metrics.diversity.base import Diversity
    >>> from brew.generation.bagging import Bagging
    >>>
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> import numpy as np
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0],
                      [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> tree = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    >>> bag = Bagging(base_classifier=tree, n_classifiers=10)
    >>> bag.fit(X, y)
    >>>
    >>> div = Diversity(metric='q')
    >>> q = div.calculate(bag.ensemble, Xtst, ytst)
    >>> q < 1.01 and q > -1.01
    True

    See also
    --------
    brew.metrics.diversity.paired: Paired diversity metrics.
    brew.metrics.diversity.non_paired: Non-paired diversity metrics.

    References
    ----------
    Brown, Gavin, et al. "Diversity creation methods: a survey and
    categorisation." Information Fusion 6.1 (2005): 5-20.

    Kuncheva, Ludmila I., and Christopher J. Whitaker. "Measures of
    diversity in classifier ensembles and their relationship with
    the ensemble accuracy." Machine learning 51.2 (2003): 181-207.

    Tang, E. Ke, Ponnuthurai N. Suganthan, and Xin Yao. "An analysis
    of diversity measures." Machine Learning 65.1 (2006): 247-271.
    """

    def __init__(self, metric='e'):
        if metric not in CLASSIFICATION_DIVERSITY_FUNCTIONS.keys():
            raise ValueError('Invalida diversity metric \'{}\'!'.format(metric))

        self.metric = CLASSIFICATION_DIVERSITY_FUNCTIONS[metric]


    def calculate(self, ensemble_oracle):
        """
        Calculates the diversity of ensemble of classifiers.

        Parameters
        ----------
        ensemble_oracle: ndarray, shape (n_samples, n_estimators)
            the return from Ensemble.oracle method.

        Returns
        -------
        diversity: float,
            Diversity of the ensemble that generated ensemble_oracle.
            Values of diversity metrics have different meanings:
            - 'e':
            - 'kw': The diversity increases with values increasing of the KW variance
            - 'q':
            - 'rho':
            - 'disagreement': The diversity increases with the value of the disagreement measure.
            - 'agreement': The diversity decreases with values increasing of the agreement measure.
            - 'df': The diversity decreases when the value of the double-fault measure increases.
        """
        if ensemble_oracle is None:
            raise ValueError('Invalid ensemble_oracle argument!')

        if ensemble_oracle.ndim != 2:
            raise ValueError('Invalid ensemble_oracle shape {}'
                    '(must be 2)!'.format(ensemble_oracle.ndim))

        if ensemble_oracle.shape[1] < 2:
            raise ValueError('Diversity requires at least 2 classifiers,'
                    'got {}'.format(oracle.shape[1]))

        return self.metric(ensemble_oracle)

