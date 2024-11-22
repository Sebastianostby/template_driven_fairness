

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import optuna

class Vectorizer:

    def __init__(self, ngram_range=(1, 1), binary=True, lowercase=True, max_features=28000, max_df=0.4, min_df=5, dtype=np.uint8) -> None:
        
        
        self.vectorizer = CountVectorizer(analyzer = 'word',
                                          ngram_range=ngram_range, 
                                          binary=binary, 
                                          lowercase=lowercase, 
                                          max_features=max_features, 
                                          max_df=max_df,
                                          min_df=min_df,
                                          dtype=dtype)

    def fit(self, documents:list):
        
        all_docs = []
        for document in documents:

            all_docs += document

        self.vectorizer.fit(all_docs)



    def transform(self, documents:list):

        if len(documents) == 1:
            self.vectorizer.vocabulary_
            return self.vectorizer.transform(documents[0]).toarray()
        
        else:
            return [self.vectorizer.transform(document).toarray() for document in documents]


    def inverse_transform(self, document:str):

        return self.vectorizer.inverse_transform(document.reshape(1, -1))


    def get_vocabulary(self):
        
        return self.vectorizer.vocabulary_
    

    def vocabulary_tester(self, words_to_check:list):

        vocab = list(self.get_vocabulary().keys())

        count_missed = 0
        for word in words_to_check:

            if word not in vocab:
                count_missed += 1
            
        captured  = len(words_to_check) - count_missed

        return captured/len(words_to_check)
    

    def create_full_vocabulary(self, documents:list, list_of_dicts:list):

        words_to_check = []
        for dict in list_of_dicts:
            for elements in dict.values():
                words_to_check.extend(elements)

        role_words = ['Nurse', 'Dr', 'Teacher', 'Professor', 'Engineer', 'Manager', 'Assistant', 'Supervisor', 'Chef', 'Officer']

        words_to_check.extend(role_words)

        def objective(trial):
            
            max_features = trial.suggest_int('max_features', 1000, 50000, step=1000)
            max_df = trial.suggest_float('max_df', 0.1, 0.8, step=0.1)
            min_df = trial.suggest_int('min_df', 1, 7, step=1)
            ngram_range_max = trial.suggest_int('ngram_range_max', 2, 10, step=1)

            self.vectorizer = CountVectorizer(analyzer = 'word',
                                              ngram_range=(1, ngram_range_max), 
                                              binary=True, 
                                              lowercase=False, 
                                              max_features=max_features, 
                                              max_df=max_df,
                                              min_df=min_df,
                                              dtype=np.uint8)

            self.fit(documents)


            return self.vocabulary_tester(words_to_check), max_features


        study = optuna.create_study(directions=['maximize', 'minimize'])

        study.optimize(objective, n_trials=500, show_progress_bar=True)

        return study.best_trials 

        



if __name__ == '__main__':


    vec = Vectorizer() 