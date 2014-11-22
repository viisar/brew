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

    def output(self, X):

        # assumes that all classifiers were trained with the same number of classes
        n_classes = len(self.get_classes())

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


