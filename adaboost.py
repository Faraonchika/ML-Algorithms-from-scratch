import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class Ada_Boost:
    def __init__(self):
        self.classifiers = []
        self.classifiers_weights = []
        self.object_weights = []
        self.errors = []
        
    
    def fit(self, X, Y, classifiers):
        n = X.shape[0]

        self.object_weights = np.zeros(shape=(classifiers, n))
        self.classifiers = np.zeros(shape=classifiers, dtype=object)
        self.classifiers_weights = np.zeros(shape=classifiers)
        self.errors = np.zeros(shape=classifiers)

        self.object_weights[0] = np.ones(shape=n) / n

        for i in range(classifiers):
            
            #Обучем пня
            curr_object_weights = self.object_weights[i]
            classifier = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            classifier = classifier.fit(X, Y, sample_weight=curr_object_weights)

            #Спрогнозируем класс, посчитаем ошибку пня и альфа тэтый
            classifier_pred = classifier.predict(X)
            err = curr_object_weights[(classifier_pred != Y)].sum()# / n
            aplha_t = np.log((1 - err) / err) / 2

            #Обновим веса объектов и отнормируем их
            new_object_weights = (curr_object_weights * np.exp(-aplha_t * Y * classifier_pred))
            new_object_weights = new_object_weights / new_object_weights.sum()

            #Для следующего пня используем уже новые веса объектов
            if i+1 < classifiers:
                self.object_weights[i+1] = new_object_weights

            self.classifiers[i] = classifier
            self.classifiers_weights[i] = aplha_t
            self.errors[i] = err
        return self

    def predict(self, X):
        #Находит предсказания каждого из пней и скалярно умножает их на полученные веса пней
        
        classifier_preds = np.array([classifier.predict(X) for classifier in self.classifiers])
        return np.sign(np.dot(self.classifiers_weights, classifier_preds))
