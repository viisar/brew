import numpy as np

from brew.metrics.diversity import paired
from brew.metrics.diversity import non_paired


class Diversity(object):
    """Ensemble Diversity Calculator.

    The class calculates the diversity of ensemble of classifiers.

    Attributes
    ----------
    `metric` : function, receives the oracle output and returns float
        Function used to calculate the metric.

    Parameters
    ----------
    metric : {'e', 'kw', 'q', 'p', 'disagreement', 'agreement', 'df'}, optional
        Metric used to compute the ensemble diversity:

        - 'e' (Entropy Measure e) will use :meth:`kuncheva_entropy_measure`
        - 'kw' (Kohavi Wolpert Variance) will use :meth:`kuncheva_kw`
        - 'q' (Q Statistics) will use :meth:`kuncheva_q_statistics`
        - 'p' (Correlation Coefficient p) will use :meth:`kuncheva_correlation_coefficient_p`  # noqa
        - 'disagreement' (Disagreement Measure) will use :meth:`kuncheva_disagreement_measure`  # noqa
        - 'agreement' (Agreement Measure) will use :meth:`kuncheva_agreement_measure` # noqa
        - 'df' (Double Fault Measure) will use :meth:`kuncheva_double_fault_measure`  # noqa

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

    def __init__(self, metric=''):
        if metric == 'e':
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
        oracle = np.equal(out, y[:, np.newaxis])

        D = self.metric(oracle)

        return D
