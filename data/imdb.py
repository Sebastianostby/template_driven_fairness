import datasets
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



def get_imdb():

    imdb = datasets.load_dataset('imdb')

    x_train, y_train, x_test, y_test = imdb['train']['text'], imdb['train']['label'], imdb['test']['text'], imdb['test']['label']

    y_train = np.array(y_train).astype(np.uint32)

    y_test = np.array(y_test).astype(np.uint32)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':

    x_train, y_train, x_test, y_test = get_imdb()

