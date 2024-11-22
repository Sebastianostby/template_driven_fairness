import green_tsetlin as gt
import numpy as np
import tqdm
import optuna

class TsetlinMachine:
    def __init__(self, n_literals:int=None, n_clauses=2000, n_classes=2, s=2.0, threshold=8000, name='tsetlin_state_c_2'):


        self.tm_name = name

        if n_literals is not None:

            self.tm = gt.TsetlinMachine(n_literals=n_literals,
                                        n_clauses=n_clauses,
                                        n_classes=n_classes,
                                        s=s,
                                        threshold=threshold)
            
        
    def train(self, x_train, y_train, x_test, y_test, n_epochs=5):

        try:
            self.tm 
            
        except:
            raise ValueError(f'deffine TM params during init!')

        trainer = gt.Trainer(self.tm, seed=42, n_jobs=-1, n_epochs=n_epochs)

        trainer.set_train_data(x_train, y_train)

        trainer.set_eval_data(x_test, y_test)

        r = trainer.train()

        return r

    def save_state(self):

        self.tm.save_state(f'models/saved_states/{self.tm_name}.npz')


    def get_predictions(self, documents, state_name=None):

        if state_name is None:
            state_name = self.tm_name + '.npz'

        try:
            tm = gt.DenseState.load_from_file(f'models/saved_states/{state_name}')

        except: 
            raise ValueError(f'Did not find tsetlin state: {state_name}')

        rs = gt.ruleset.RuleSet(is_multi_label=False)

        rs.compile_from_dense_state(tm)

        predictor = gt.Predictor(explanation='none', multi_label=False)
        predictor._set_ruleset(rs)
        predictor._allocate_backend()

        return np.array([predictor.predict(instance) for instance in tqdm.tqdm(documents, desc=f'classifying documents using {state_name}')], dtype=np.uint32)





