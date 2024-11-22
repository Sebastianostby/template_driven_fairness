

import json
import re
from sklearn.feature_extraction.text import CountVectorizer
import datasets
import numpy as np


def get_template_driven_fairness_data():

    with open('data/raw_files/templates.json', 'rb') as f:
        templates = json.load(f)

    with open('data/raw_files/descriptive_words.json', 'rb') as f:
        descriptives = json.load(f)

    with open('data/raw_files/gender_names.json', 'rb') as f:
        objects = json.load(f)
    
    male_names = objects['male']

    female_names = objects['female']

    text = []
    labels = []
    attribute = []

    for instance in templates:

        template = instance['template']

        for label, descriptive in descriptives.items():
            
            for word in descriptive:

                template_with_descriptive = re.sub(r'<descriptive word>', word, template)

                for male, female in zip(male_names, female_names):
                    
                    text.append(re.sub(r'<object>', male, template_with_descriptive))
                    text.append(re.sub(r'<object>', female, template_with_descriptive))
                
                    labels.append(1 if label == "positive" else 0)
                    labels.append(1 if label == "positive" else 0)

                    attribute.append('male')
                    attribute.append('female')


    print(f'Generated #{len(text)} documents.')
    return text, np.array(labels, dtype=np.uint32), attribute





if __name__ == '__main__':

    text, lables, attribute = get_template_driven_fairness_data()

    for n, (i, j, k) in enumerate(zip(text, lables, attribute)):
        print(i, j, k)

        if n == 10:
            break
