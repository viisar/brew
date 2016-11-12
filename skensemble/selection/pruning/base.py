
class Prunner(object):

    def __init__(self):
        pass

    def fit(self, ensemble, X, y):
        return self

    def get(self, p=0.1):
        return self.ensemble[:int(p * len(self.ensemble))]
