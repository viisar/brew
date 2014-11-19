import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import sklearn.datasets as datasets
import combination.rules as rules



def transform2votes(output):

    # TODO: it's getting the number of classes
    # using the predictions, this can FAIL
    # if there is a class which is never
    # predicted, fix LATER

    n_samples = output.shape[0]
    n_classes = np.unique(output).shape[0]

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

    def output(self, X):

        out = []

        for i, c in enumerate(self.classifiers):

            #if mode == 'vote':
            tmp = c.predict(X) # [n_samples, 1]
            out.append(transform2votes(tmp))

            #elif mode == 'prob':
            #    out.append(c.predict_proba(X)) # [n_samples, n_classes] 
        # out = [n_samples, n_classes, n_classifiers]

        out = np.array(out)
        
        return out.reshape((out.shape[1], out.shape[2], out.shape[0]))

    def __len__(self):
        return len(self.classifiers)


class EnsembleClassifier(object):

    def __init__(self, ensemble=None, combiner=None):
        self.ensemble = ensemble

        if combiner != rules.majority_vote:
            raise Exception("Use majority voting for the time being!")

        self.combiner = combiner

    def predict(X):

        # TODO: warn the user if mode of ensemble
        # output excludes the chosen combiner
        out = self.ensemble.output(X, mode='vote')
        y = self.combiner(out)

        return y


def load_iris():
    iris = datasets.load_iris()
    data = iris['data']
    target = iris['target']

    dataset = np.concatenate((data, target.reshape((150,1))), axis=1)

    # shuffle dataset
    np.random.shuffle(dataset)

    # train_set, valid_set, test_set
    train_set = dataset[:105, :]    # 70%
    test_set = dataset[105:, :]  # 30%

    return train_set, test_set



if __name__ == '__main__':

    pool = Ensemble()

    c1 = SVC()

    train_set, test_set = load_iris()

    X_train = train_set[:,:-1]
    y_train = train_set[:,-1:].ravel()

    X_test = test_set[:,:-1]
    y_test = test_set[:,-1:].ravel()

    c1.fit(X_train, y_train)

    pool.add(c1)
    
    out = pool.output(X_test)

    print(out.shape)

    print(rules.majority_vote_rule(out))

