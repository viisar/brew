import sklearn


class Evaluator(object):
    
    def __init__(self, metric='auc'):
        if metric == 'auc':
            self.metric = auc_score
        elif metric == 'acc':
            self.metric = acc_score
        elif metric == 'avg':
            
            


def auc_score(y_true, y_pred, positive_label=1):
    fp_rate, tp_rate, thresholds = sklearn.metrics.roc_curve(
        y_true, y_pred, pos_label=positive_label)
    return sklearn.metrics.auc(fp_rate, tp_rate)

def acc_score(y_true, y_pred, positive_label=1):
    return float(np.sum(y_true == y_pred))/y_true.shape[0]

def gmean_score(y_true, y_pred, positive_label=1):


def avg_score(y_true, y_pred, positive_label=1):
    return sklearn.metrics.auc(fp_rate, tp_rate)


