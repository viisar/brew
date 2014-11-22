import numpy as np

from brew.combination.combiner import Combiner


def transform2votes(output, n_classes):

    # TODO: it's getting the number of classes
    # using the predictions, this can FAIL
    # if there is a class which is never
    # predicted, fix LATER

    n_samples = output.shape[0]
    #n_classes = np.unique(output).size

    votes = np.zeros((n_samples, n_classes))

    # uses the predicted label as index for the vote matrix
    for i in range(n_samples):
        idx = output[i]
        votes[i, idx] = 1

    #if np.sum(votes.sum(axis=1)) != n_samples

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

    def output(self, X, n_classes=2):

        out = np.zeros((X.shape[0], n_classes, len(self.classifiers)))

        for i, c in enumerate(self.classifiers):
            tmp = c.predict(X) # (n_samples,)
            votes = transform2votes(tmp, n_classes) # (n_samples, n_classes)
            out[:,:,i] = votes

        return out



    def __len__(self):
        return len(self.classifiers)


class EnsembleClassifier(object):

    def __init__(self, ensemble=None, combiner=None):
        self.ensemble = ensemble

        if combiner == None:
            combiner = Combiner(rule='majority_vote')
        
        self.combiner = combiner

    def predict(self, X):

        # TODO: warn the user if mode of ensemble
        # output excludes the chosen combiner?

        out = self.ensemble.output(X)
        y = self.combiner.combine(out)

        return y


