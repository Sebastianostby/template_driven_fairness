import numpy as np
from fairness_frame.equal_odds import get_equal_odds
from fairness_frame.accuracy import get_accuracy 
from fairness_frame.demographic_parity import get_demographic_parity
from fairness_frame.paird_t_test import get_t_test
from dependencies.json_encoder import NumpyEncoder
import json

class FairnessFrame:
    def __init__(self, test_name=f'random_test') -> None:
        
        self.test_name = test_name 

        self.fairness_frame = {'test_name': self.test_name}
       
        self.results = None

    def reset(self):

        self.fairness_frame = {'test_name': self.test_name}

        self.results = None


    def set_data(self, labels, predictions, masks):

        unique_groups = list(set(masks))
        
        labels = np.asarray(labels) if not isinstance(labels, np.ndarray) else labels
        predictions = np.asarray(predictions) if not isinstance(predictions, np.ndarray) else predictions
        masks = np.asarray(masks) if not isinstance(masks, np.ndarray) else masks
        

        self.fairness_frame['group_space'] = unique_groups
        self.fairness_frame['labels'] = labels
        self.fairness_frame['predictions'] = predictions
        self.fairness_frame['masks'] = masks

        for i, group in enumerate(unique_groups):
            
            group_name = str(group)

            group_instances = np.where(masks==group_name)
            group_predictions = predictions[group_instances]
            group_labels = labels[group_instances]

            self.fairness_frame[group_name] = {
                'group_shift' : i + 1,
                'n_instances' : len(group_instances[0]),
                'group_instances' : group_instances,
                'group_predictions' : group_predictions,
                'group_labels' : group_labels
            }


    def calculate_fairness(self, explainer_data:list=None):

        self.results = {}

        for group in self.fairness_frame['group_space']:
            self.results[group] = {}

        self.results = get_accuracy(frame=self.fairness_frame, results=self.results)
        self.results = get_demographic_parity(frame=self.fairness_frame, results=self.results)
        self.results = get_equal_odds(frame=self.fairness_frame, results=self.results) 
        self.results = get_t_test(frame=self.fairness_frame, results=self.results) 


        if explainer_data is not None:
            self.explainer(explainer_data)

        return self.results


    def explainer(self, texts):

        unique_groups = self.fairness_frame['group_space']
    
        for group in unique_groups:
            predictions = self.fairness_frame[group]['group_predictions']
            labels = self.fairness_frame[group]['group_labels']
            instances = self.fairness_frame[group]['group_instances'][0]
            
            errors = np.where(labels != predictions)[0]
            
            for error in errors:
                error_index = instances[error]
                error_text = texts[error_index]
                
                print(f"Error in group {group}:")
                print(f"  Text: {error_text}")
                print(f"  Prediction: {predictions[error]}")
                print(f"  Actual Label: {labels[error]}")
                
                # Find corresponding instance in other group
                other_group = [g for g in unique_groups if g != group][0]
                other_instances = self.fairness_frame[other_group]['group_instances'][0]
                other_predictions = self.fairness_frame[other_group]['group_predictions']
                other_labels = self.fairness_frame[other_group]['group_labels']
                
                corresponding_index = other_instances[error]
                corresponding_text = texts[corresponding_index]
                corresponding_prediction = other_predictions[error]
                corresponding_label = other_labels[error]
                
                print(f"Corresponding instance in group {other_group}:")
                print(f"  Text: {corresponding_text}")
                print(f"  Prediction: {corresponding_prediction}")
                print(f"  Actual Label: {corresponding_label}")
                print()


            


        



    def save_file(self):
        

        if self.results is None:
            raise ValueError('No results to save...')

        
        with open(f'outputs/{self.test_name}.json', 'w') as f:
            json.dump(self.results, f, indent=4, cls=NumpyEncoder)


        print(f'file wrote to {self.test_name}.json')
        


