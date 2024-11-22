import numpy as np
from sklearn.metrics import accuracy_score


def get_accuracy(frame, results):

    results['global_accuracy'] = accuracy_score(frame['labels'], frame['predictions'])

    for group in frame['group_space']:

        predictions = frame[group]['group_predictions']

        labels = frame[group]['group_labels']

        results[group]['group_accuracy'] = accuracy_score(labels, predictions)

    return results