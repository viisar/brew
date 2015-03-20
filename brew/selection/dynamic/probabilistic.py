



class Priori(DCS):

    @abstractmethod
    def probability(self, clf, nn_X, nn_y):
        clf.predict_proba(nn_X)

    def select(self, ensemble, x):
        nn_X, nn_y, dists = self.get_neighbors(x, 
                return_distance=True)
        
        selected_id = []
        selected_pb = []
        
        for idx, clf in enumerate(ensemble.classifiers):
            prob = self.probability(nn_X, nn_y)
            if prob > self.threshold:
                selected_id = selected_id + [idx]
                selected_pb = selected_pb + [prob]
        

        return None, None



class Posteriori(DCS):

    def select(self, ensemble, x):
        pass

