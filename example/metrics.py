from collections import Counter
import sklearn.metrics as metrics

"""
    This script contains a series of metrics that can be extracted from any machine learning algorithms
    Mostly of the methods receive the <y_true> that is the gold and <y_pred> which is the algorithm predictions.
"""


def metric_accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


def metric_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average="weighted")


def metric_f1_micro(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average="micro")


def metric_f1_macro(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average="macro")


def metric_precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, average="binary")


def metric_recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, average="binary")


def metric_kappa(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred)


def metric_roc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred, average="weighted")


def metric_true_positive(y_true, y_pred):
    _tn, _fp, _fn, tp = _confusion_matrix(y_true, y_pred)
    return tp


def metric_true_negative(y_true, y_pred):
    tn, _fp, _fn, _tp = _confusion_matrix(y_true, y_pred)
    return tn


def metric_false_positive(y_true, y_pred):
    _tn, fp, _fn, _tp = _confusion_matrix(y_true, y_pred)
    return fp


def metric_false_negative(y_true, y_pred):
    _tn, _fp, fn, _tp = _confusion_matrix(y_true, y_pred)
    return fn


def metric_positive_predictive_value(y_true, y_pred):
    _tn, fp, _fn, tp = _confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def metric_negative_predictive_value(y_true, y_pred):
    tn, _fp, fn, _tp = _confusion_matrix(y_true, y_pred)
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def metric_sensitivity(y_true, y_pred):
    _tn, _fp, fn, tp = _confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def metric_specificity(y_true, y_pred):
    tn, fp, _fn, _tp = _confusion_matrix(y_true, y_pred)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def metric_diff_expected(y_true, y_pred):
    return abs(Counter(y_true)[1] - Counter(y_pred)[1])


def _confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return int(tn), int(fp), int(fn), int(tp)
