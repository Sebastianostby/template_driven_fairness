
from itertools import combinations
from scipy.stats import ttest_rel



def get_t_test(frame, results):

    for g1, g2 in combinations(frame['group_space'], 2):

        g1_pred = frame[g1]['group_predictions']
        g2_pred = frame[g2]['group_predictions']

        t_stat, p_value = ttest_rel(g1_pred, g2_pred)

        results[f'{g1}-{g2}']['t_test_p_value'] = p_value

    return results

            
