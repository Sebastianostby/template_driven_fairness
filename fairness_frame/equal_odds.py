
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, equalized_odds_difference
from itertools import combinations


def get_equal_odds(frame, results):

    y_true = frame['labels']
    y_pred = frame['predictions']
    sensitive_features = frame['masks']

    metrics = {'tpr': true_positive_rate, 
               'fpr': false_positive_rate}

    mf = MetricFrame(metrics=metrics,
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_features)



    for group in frame['group_space']:

        results[group]['tpr'] = mf.by_group['tpr'][group]
        results[group]['fpr'] = mf.by_group['fpr'][group]



    max_tpr = 0
    max_fpr = 0
    for g1, g2, in combinations(frame['group_space'], 2):
       
        tpr_diff = abs(results[g1]['tpr'] - results[g2]['tpr'])
        fpr_diff = abs(results[g1]['fpr'] - results[g2]['fpr'])
        
        max_tpr = max(max_tpr, tpr_diff)
        max_fpr = max(max_fpr, fpr_diff)
       

        results[f'{g1}-{g2}'] = {
                'tpr_difference': tpr_diff,                
                'fpr_difference': fpr_diff
                }


    results['max_fpr_difference'] = max_fpr
    results['equal_opportunity'] = max_tpr
    results['eo'] = max(max_tpr, max_fpr)

    return results
