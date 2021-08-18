import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize


def loss_func(y, y_pred): 
    return ((y - y_pred)**2) / 2

def loss_direction(y, y_pred): 
    return -(y-y_pred)

def XGBoost(X, y, T): 
    history = []
    composition = []
    a_array = []
    
    #Инициализируем первый алгоритм (тут среднее значение)
    y_pred = np.array([y.mean()] * len(y)) 
    a0 = y_pred 
    history.append(loss_func(y, y_pred).mean())
    for t in range(T): 
        #Вычисляем псевдоостатки
        residuals = -loss_direction(y, y_pred) 
        
        #Обучаем алгоритм bt
        bt = DecisionTreeRegressor(max_depth=2) 
        bt.fit(X, residuals) 
        composition.append(bt) 
        predictions = bt.predict(X) 
        
        #Найдем at путём минимизации
        at = minimize(lambda at: (((y - (y_pred + at * predictions))**2) / 2).mean(), [0.1]).x[0]
        a_array.append(at)
        
        #Добавим алгоритм в композицию
        y_pred = y_pred + at * predictions 
        history.append(loss_func(y, y_pred).mean()) 
        
    return composition, a0, a_array, history

def XGBoost_predict(X, composition, a0, a_array):
    for t in range(len(composition)): 
        y_pred = a0 + a_array[t] * composition[t].predict(X) 
    return y_pred
