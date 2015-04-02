import numpy as np

from brew.combination.combiner import Combiner
from brew.metrics.evaluation import auc_score


def transform2votes(output, n_classes):

    n_samples = output.shape[0]

    votes = np.zeros((n_samples, n_classes))

    # uses the predicted label as index for the vote matrix
    for i in range(n_samples):
        idx = output[i]
        votes[i, idx] = 1

    return votes.astype('int')


class Ensemble(object):
    """Class that represents a collection of classifiers.

    The Ensemble class serves as a wrapper for a list of classifiers,
    besides providing a simple way to calculate the output of all the
    classifiers in the ensemble. 
    
    Attributes
    ----------
    `classifiers` : list 
        Stores all classifiers in the ensemble.

    `yval` : array-like, shape = [indeterminated]
        Labels of the validation set.

    `knn`  : sklearn KNeighborsClassifier,
        Classifier used to find neighborhood.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> from brew.base import Ensemble
    >>>
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0] , [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>>
    >>> dt1 = DecisionTreeClassifier()
    >>> dt2 = DecisionTreeClassifier()
    >>>
    >>> dt1.fit(X, y)
    >>> dt2.fit(X, y)
    >>>
    >>> ens = Ensemble(classifiers=[dt1, dt2])
 
    """
    def __init__(self, classifiers=None):
        
        if classifiers == None:
            self.classifiers = []
        else:
            self.classifiers = classifiers

    def add(self, classifier):
        self.classifiers.append(classifier)

    def add_classifiers(self, classifiers):
        self.classifiers = self.classifiers + classifiers

    def add_ensemble(self, ensemble):
        self.classifiers = self.add_classifiers(ensemble.classifiers)

    def get_classes(self):
        classes = set()
        for c in self.classifiers:
            classes = classes.union(set(c.classes_))

        self.classes_ = list(classes)
        return self.classes_

    def output(self, X, mode='votes'):
        """Returns the output of all classifiers packed in a numpy array.
       
        This method calculates the output of each classifier, and stores them
        in a array-like shape. The specific shape and the meaning of each element
        is defined by argument `mode`. 

        (1) 'labels': each classifier will return a single label prediction
        for each sample in X, therefore the ensemble output will be a 2d-array
        of shape (n_samples, n_classifiers), with elements being the class labels.

        (2) 'probs': each classifier will return the posterior probabilities of each
        class (i.e. instead of returning a single choice it will return the
        probabilities of each class label being the right one). The ensemble output
        will be a 3d-array with shape (n_samples, n_classes, n_classifiers), with
        each element being the probability of a specific class label being right on a
        given sample according to one the classifiers. This mode can be used with
        any combination rule.

        (3) 'votes': each classifier will return votes for each class label (i.e.
        a binary representation, where the chosen class label will have one vote
        and the other labels will have zero votes. The ensemble output will be
        a binary 3d-array with shape (n_samples, n_classes, n_classifiers), with
        the elements being the votes. This mode is mainly used in combining the
        classifiers output by using majority vote rule.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
                The test input samples.

        mode: string, optional(default='labels')
                The type of output given by each classifier.  
                'labels' | 'probs' | 'votes'
            
            
        """

        if mode == 'labels':
            out = np.zeros((X.shape[0], len(self.classifiers)))
            for i, clf in enumerate(self.classifiers):
                out[:,i] = clf.predict(X)

        else:
            # assumes that all classifiers were
            # trained with the same number of classes
            n_classes = len(self.get_classes())
            out = np.zeros((X.shape[0], n_classes, len(self.classifiers)))

            for i, c in enumerate(self.classifiers):
                if mode == 'probs':
                    tmp = c.predict_proba(X)
                    out[:,:,i] = tmp

                elif mode == 'votes':
                    tmp = c.predict(X) # (n_samples,)
                    votes = transform2votes(tmp, n_classes) # (n_samples, n_classes)
                    out[:,:,i] = votes

        return out

    def output_simple(self, X):
        out = np.zeros((X.shape[0], len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            out[:,i] = clf.predict(X)

        return out


    def in_agreement(self, x):
        prev = None
        for clf in self.classifiers:
            tmp = clf.predict(x)
            if tmp != prev:
                return False
            prev = tmp

        return True

    def __len__(self):
        return len(self.classifiers)


class EnsembleClassifier(object):

    def __init__(self, ensemble=None, selector=None, combiner=None):
        self.ensemble = ensemble
        self.selector = selector
                
        if combiner == None:
            combiner = Combiner(rule='majority_vote')

        self.combiner = combiner

    def predict(self, X):

        # TODO: warn the user if mode of ensemble
        # output excludes the chosen combiner?

        if self.selector == None:
            out = self.ensemble.output(X)
            y = self.combiner.combine(out)


        else:
            y = []

            for i in range(X.shape[0]):
                ensemble, weights = self.selector.select(self.ensemble, X[i,:][np.newaxis,:])
                    
                if weights is not None: # use the ensemble with weights
                    out = ensemble.output(X[i,:][np.newaxis,:])
                    
                    # apply weights
                    for i in range(out.shape[2]):
                        out[:,:,i] = out[:,:,i] * weights[i]

                    [tmp] = self.combiner.combine(out)
                    y.append(tmp)
                    
                else: # use the ensemble, but ignore the weights
                    out = ensemble.output(X[i,:][np.newaxis,:])
                    [tmp] = self.combiner.combine(out)
                    y.append(tmp)

        return np.asarray(y)


def oracle(ensemble, y_true, metric=auc_score):
    out = ensemble.output(X, mode='labels')
    oracle = np.equal(out, y[:,np.newaxis])
    mask = np.any(oracle, axis=1)
    y_pred = ensemble_output[:,0]
    y_pred[mask] = y_true
    return metric(y_pred, y_true)

def single_best(ensemble, y_true, metric=auc_score):
    out = ensemble.output(X, mode='labels')
    scores = metric(out, y_true[:,np.newaxis])
    return np.max(scores)







