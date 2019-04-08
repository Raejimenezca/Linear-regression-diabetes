# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:36:54 2019

@author: Usuario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

dbDiabetes = pd.read_csv('diabetes.csv')
print(dbDiabetes.head())

print(dbDiabetes.sort_values('age')['age'])
#
#for item in ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']:
#    X = dbDiabetes[[item]]
#    Y = dbDiabetes['Y']
#    
#    dbDiabetes.sort_values(item)[item]
#    
#    lm = LinearRegression()
#    print(item, cross_val_score(lm, X, Y, cv=20).mean())
#    

plt.figure(0)
plt.plot(dbDiabetes.sort_values('s6')['s6'],range(len(dbDiabetes.sort_values('age')['age'])))

plt.figure(2)
plt.plot(dbDiabetes.sort_values('Y')['Y'],range(len(dbDiabetes.sort_values('age')['age'])))

plt.show()