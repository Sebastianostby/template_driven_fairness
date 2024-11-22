from data import imdb
from data import tdfd

from fairness_frame.fairness_frame import FairnessFrame

from dependencies.vectorizer import Vectorizer
from sklearn.tree import DecisionTreeClassifier

def fairness_experiment():


    x_train, y_train, x_test, y_test =  imdb.get_imdb()

    fairness_text, fairness_labels, protected_attributes = tdfd.get_template_driven_fairness_data()

    vectorizer = Vectorizer(max_features=3000, max_df=0.8, min_df=5, ngram_range=(1, 4))

    vectorizer.fit([x_train, x_test, fairness_text])
    
    x_train, x_test, text_vec = vectorizer.transform([x_train, x_test, fairness_text])

    model = DecisionTreeClassifier()

    model.fit(x_train, y_train)

    ff = FairnessFrame()

    ff.set_data(predictions=model.predict(text_vec), labels=fairness_labels, masks=protected_attributes)

    results = ff.calculate_fairness()
 
    print(results)