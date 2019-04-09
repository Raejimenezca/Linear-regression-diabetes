# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:36:54 2019

@author: Usuario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
#from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.metrics import mean_squared_error
#from sklearn.neighbors import KNeighborsRegressor


## leer de las columnas
colnames = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'y']

## Leer el archivo CSV
dbDiabetes = pd.read_csv('diabetes.csv', names = colnames)

# Pasar columnas que nos interesan del CSV a lista
X = dbDiabetes.s6.tolist()
Y = dbDiabetes.y.tolist()

## Elimina el retorno de carro de los strings
##dbDiabetes = [x.split(',') for x in dbDiabetes]



X = [float(y) for y in X[1:]]
Y = [float(z) for z in Y[1:]]
#print(X)
#print(Y)



## Convierte todas las columnas a float
##dbDiabetes = [[float(y) for y in x] for x in dbDiabetes]

#for item in ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']:
#    X = dbDiabetes[[item]]
#    Y = dbDiabetes['Y']
#    
#    dbDiabetes.sort_values(item)[item]
#    
#    lm = LinearRegression()
#    print(item, cross_val_score(lm, X, Y, cv=20).mean())
    

# partimos del hecho que solo 2 variables son relevantes para el modelo, s6 = glucosa y 'Y' el nivel de azucar en la sangre 
#plt.figure(0)
#plt.plot(dbDiabetes.sort_values('s6')['s6'],range(len(dbDiabetes.sort_values('age')['age'])))
#
#plt.figure(2)
#plt.plot(dbDiabetes.sort_values('Y')['Y'],range(len(dbDiabetes.sort_values('age')['age'])))
#
#plt.show()

m = linear_model.LinearRegression()
m.fit(np.array(X).reshape(-1, 1), Y)
y_pred = m.predict(np.array(X).reshape(-1, 1))
y_pred
plt.figure(3)
plt.plot(X, Y, '.r')
plt.plot(X, y_pred, '-b')

