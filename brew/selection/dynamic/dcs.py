
class OLA(DCS):

    def predict(self, X):
        X_tst = np.array(X)
        y_pred = []
        for i in range(X_tst.shape[0]):
            if self.agreement(X_tst[i]):
                y_pred += [self.classifiers[0].predict(X_tst[i])]
            else:
                [idx] = self.knn.kneighbors(X_tst[i], return_distance=False)
                scores = [clf.score(self.val_X[idx], self.val_y[idx]) for clf in self.classifiers]
                clf = self.classifiers[np.argmax(scores)]
                y_pred += [clf.predict(X_tst[i])]

        return np.array(y_pred)

class OLA2(DCS):

    def predict(self, X):
        X_tst = np.array(X)

        X_idxs = self.knn.kneighbors(X_tst, return_distance=False)
        c_idxs = map(lambda e: np.argmax([clf.score(self.val_X[e], self.val_y[e]) for clf in self.classifiers]), X_idxs)
        y_pred = [self.classifiers[idx].predict(X_tst[i]) for (idx, i) in zip(c_idxs, range(X_tst.shape[0]))]

        return np.asarray(y_pred)

class LCA(DCS):

    def predict(self, X):
        X_tst = np.array(X)
        y_pred = []

        for i in range(X_tst.shape[0]):
            if self.agreement(X_tst[i]):
                y_pred += [self.classifiers[0].predict(X_tst[i])]
            else:
                [idx] = self.knn.kneighbors(X_tst[i], return_distance=False)
                mx_id, mx_vl = -1, -1
                for e, clf in enumerate(self.classifiers):
                    right, count = 0, 0
                    for xv, yv in zip(self.val_X[idx], self.val_y[idx]):
                        pred = clf.predict(xv)
                        if pred == clf.predict(X_tst[i]):
                            count = count + 1
                            if pred == yv:
                                right = right + 1
                    if right > 0 and count > 0 and float(right)/count > mx_vl:
                        mx_id, mx_vl = e, float(right)/count
                        
                y_pred += [self.classifiers[mx_id].predict(X_tst[i])]

        return np.array(y_pred)


 

 

