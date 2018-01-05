import numpy as np


def metric_accuracy(predictions, labels):
    """
    Function for calculating the accuracy.

    :param predictions:
    :param labels:
    :return:
        accuracy: float, the accuracy
        true_count: int,
        total_count: int,
    """
    assert predictions.shape == labels.shape, \
        'shape of predictions%s not equal to labels%s' % (predictions.shape, labels.shape)
    correct = (predictions == labels)
    true_count = int(np.sum(correct))
    total_count = int(np.prod(correct.shape))
    accuracy = true_count * 1.0 / total_count
    return accuracy, true_count, total_count


def count_many(predictions, labels):
    """
    Function that return many counter, including TP, TN, FP, FN.
    Labels are assumed to have only two classes, namely 0 and 1.

    :param predictions:
    :param labels:
    :return:
        result: dict
    """
    assert predictions.shape == labels.shape, \
        'shape of predictions%s not equal to labels%s' % (predictions.shape, labels.shape)
    assert (set(np.unique(predictions)) | set(np.unique(labels))) == {0, 1}, \
        '<<ERROR>> classes set not equal {0, 1}'

    correct = (predictions == labels)
    wrong = (predictions != labels)
    true_positives = int(np.sum(correct & (labels == 1)))
    true_negatives = int(np.sum(correct & (labels == 0)))
    false_positives = int(np.sum(wrong & (predictions == 1)))
    false_negatives = int(np.sum(wrong & (predictions == 0)))
    result = {
        'TP': true_positives,
        'TN': true_negatives,
        'FP': false_positives,
        'FN': false_negatives,
    }
    return result


def metric_many_from_counter(result):
    TP, TN, FP, FN = result['TP'], result['TN'], result['FP'], result['FN']
    accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN)
    sensitivity = TP * 1.0 / (TP + FN)
    specificity = TN * 1.0 / (TN + FP)
    out = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    return out


def metric_many_from_predictions(predictions, labels):
    result = count_many(predictions, labels)
    return metric_many_from_counter(result)
