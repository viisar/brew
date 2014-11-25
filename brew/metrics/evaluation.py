import sklearn

def auc_score(y_true, y_pred, positive_label=1):
    fp_rate, tp_rate, thresholds = sklearn.metrics.roc_curve(
        y_true, y_pred, pos_label=positive_label)
    return sklearn.metrics.auc(fp_rate, tp_rate)
