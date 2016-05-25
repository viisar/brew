=============================
brew
=============================

.. image:: https://badges.gitter.im/viisar/brew.svg
   :alt: Join the chat at https://gitter.im/viisar/brew
   :target: https://gitter.im/viisar/brew?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://badge.fury.io/py/brew.png
    :target: http://badge.fury.io/py/brew

.. image:: https://travis-ci.org/viisar/brew.png?branch=master
    :target: https://travis-ci.org/viisar/brew

.. image:: https://pypip.in/d/brew/badge.png
    :target: https://pypi.python.org/pypi/brew

.. image:: https://pypip.in/d/brew/badge.png
    :target: https://testpypi.python.org/pypi/brew


BREW: A Multiple Classifier Systems API

This project was started in 2014 by Dayvid Victor and Thyago Porpino for the project of the Multiple Classifier Systems class at Federal University of Pernambuco.

The aim of this project is to provide a structure for Ensemble Generation, Ensemble Pruning, and Static and Dynamic selection of classifiers.


Dependencies
============
- Python 2.6+
- scikit-learn >= 0.14.1
- Numpy >= 1.3
- SciPy >= 0.7
- Matplotlib >= 0.99.1 (for examples, only)

Features
--------
* Dynamic Classifier Selection: OLA, LCA, A Priori, A Posteriori.
* Dynamic Ensemble Selection: KNORA E and KNORA U.
* Ensemble Combination Rules: majority vote, min, max, mean and median.
* Ensemble Diversity Metrics: Entropy Measure E, Kohavi Wolpert Variance, Q Statistics, Correlation Coefficient p, Disagreement Measure, Agreement Measure, Double Fault Measure.
* Ensemble Classifier Generators: Bagging, Random Subspace, SMOTEBagging, ICS-Bagging, SMOTE-ICS-Bagging.
* Ensemble Pruning: EPIC.
* Oversampling: SMOTE.


Important References
====================

- Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2014.

- Zhou, Zhi-Hua. Ensemble methods: foundations and algorithms. CRC Press, 2012.

