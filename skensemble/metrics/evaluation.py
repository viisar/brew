import numpy as np

import sklearn
import sklearn.metrics


class Evaluator(object):

    def __init__(self, metric='auc'):
        if metric == 'auc':
            self.metric = auc_score
        elif metric == 'acc':
            self.metric = acc_score

    def calculate(self, y_true, y_pred):
        return self.metric(y_true, y_pred)


def auc_score(y_true, y_pred, positive_label=1):
    if hasattr(sklearn.metrics, 'roc_auc_score'):
        return sklearn.metrics.roc_auc_score(y_true, y_pred)

    fp_rate, tp_rate, thresholds = sklearn.metrics.roc_curve(
        y_true, y_pred, pos_label=positive_label)
    return sklearn.metrics.auc(fp_rate, tp_rate)


def acc_score(y_true, y_pred, positive_label=1):
    if hasattr(sklearn.metrics, 'accuracy_score'):
        return sklearn.metrics.accuracy_score(y_true, y_pred)

    return float(np.sum(y_true == y_pred)) / y_true.shape[0]
