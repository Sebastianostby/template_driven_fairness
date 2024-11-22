
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from itertools import combinations


def get_demographic_parity(frame, results):

    y_true = frame['labels']
    y_pred = frame['predictions']
    sensitive_features = frame['masks']


    max_demographic_parity = demographic_parity_difference(
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_features)


    results['max_demographic_parity_difference'] = max_demographic_parity

    return results
