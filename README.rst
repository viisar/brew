=============================
brew
=============================

.. image:: https://badge.fury.io/py/brew.png
    :target: http://badge.fury.io/py/brew

.. image:: https://travis-ci.org/viisar/brew.png?branch=master
    :target: https://travis-ci.org/viisar/brew

.. image:: https://landscape.io/github/viisar/brew/master/landscape.svg?style=flat
   :target: https://landscape.io/github/viisar/brew/master
   :alt: Code Health

.. image:: https://coveralls.io/repos/github/viisar/brew/badge.svg?branch=master
    :target: https://coveralls.io/github/viisar/brew?branch=master

.. image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Join the chat at https://gitter.im/viisar/brew
   :target: https://gitter.im/viisar/brew?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge


**brew: A Multiple Classifier Systems API**

| This project was started in 2014 by *Dayvid Victor* and *Thyago Porpino*
| for the Multiple Classifier Systems class at Federal University of Pernambuco.


| The aim of this project is to provide an easy API for Ensembling, Stacking, 
| Blending, Ensemble Generation, Ensemble Pruning, Dynamic Classifier Selection, 
| and Dynamic Ensemble Selection.

Features
===========
* General: Ensembling, Stacking and Blending.
* Ensemble Classifier Generators: Bagging, Random Subspace, SMOTE-Bagging, ICS-Bagging, SMOTE-ICS-Bagging.
* Dynamic Selection: Overall Local Accuracy (OLA), Local Class Accuracy (LCA), Multiple Classifier Behavior (MCB), K-Nearest Oracles Eliminate (KNORA-E), K-Nearest Oracles Union (KNORA-U), A Priori Dynamic Selection, A Posteriori Dynamic Selection, Dynamic Selection KNN (DSKNN).
* Ensemble Combination Rules: majority vote, min, max, mean and median.
* Ensemble Diversity Metrics: Entropy Measure E, Kohavi Wolpert Variance, Q Statistics, Correlation Coefficient p, Disagreement Measure, Agreement Measure, Double Fault Measure.
* Ensemble Pruning: Ensemble Pruning via Individual Contribution (EPIC).

Example
============

.. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import itertools

        import sklearn

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier

        from brew.base import Ensemble, EnsembleClassifier
        from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
        from brew.combination.combiner import Combiner

        from mlxtend.data import iris_data
        from mlxtend.evaluate import plot_decision_regions

        # Initializing Classifiers
        clf1 = LogisticRegression(random_state=0)
        clf2 = RandomForestClassifier(random_state=0)
        clf3 = SVC(random_state=0, probability=True)

        # Creating Ensemble
        ensemble = Ensemble([clf1, clf2, clf3])
        eclf = EnsembleClassifier(ensemble=ensemble, combiner=Combiner('mean'))

        # Creating Stacking
        layer_1 = Ensemble([clf1, clf2, clf3])
        layer_2 = Ensemble([sklearn.clone(clf1)])

        stack = EnsembleStack(cv=3)

        stack.add_layer(layer_1)
        stack.add_layer(layer_2)

        sclf = EnsembleStackClassifier(stack)

        clf_list = [clf1, clf2, clf3, eclf, sclf]
        lbl_list = ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble', 'Stacking']

        # Loading some example data
        X, y = iris_data()
        X = X[:,[0, 2]]

        # Plotting Decision Regions
        gs = gridspec.GridSpec(2, 3)
        fig = plt.figure(figsize=(10, 8))

        itt = itertools.product([0, 1, 2], repeat=2)

        for clf, lab, grd in zip(clf_list, lbl_list, itt):
            clf.fit(X, y)
            ax = plt.subplot(gs[grd[0], grd[1]])
            fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
            plt.title(lab)
        plt.show()


.. image:: https://raw.githubusercontent.com/viisar/brew/master/docs/sources/img/iris_decision_regions_2d.png
    :alt: decision regions plots
    :align: center


Dependencies
============
- Python 2.7+
- scikit-learn >= 0.15.2
- Numpy >= 1.6.1
- SciPy >= 0.9
- Matplotlib >= 0.99.1 (examples, only)
- mlxtend (examples, only)


Installing
==========

You can easily install brew using ``pip``::

    pip install brew

or, if you prefer an up-to-date version, get it from here::

    pip install git+https://github.com/viisar/brew.git


Important References
====================

- Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2014.
- Zhou, Zhi-Hua. Ensemble methods: foundations and algorithms. CRC Press, 2012.

