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

