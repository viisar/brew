import numpy as np

from brew.combination.combiner import Combiner


def transform2votes(output, n_classes):

    n_samples = output.shape[0]

    votes = np.zeros((n_samples, n_classes))

    # uses the predicted label as index for the vote matrix
    for i in range(n_samples):
        idx = output[i]
        votes[i, idx] = 1

    return votes.astype('int')


class Ensemble(object):

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
