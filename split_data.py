import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import numpy as np
pd.options.mode.chained_assignment = None

dataset = pd.read_csv('features.csv')
y = dataset.pop('class')
X = dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y)

X_train['class'] = y_train

X_test['class'] = y_test

min_max_scaler = preprocessing.MinMaxScaler()
colunas = X_train.columns[1:-1]
X_train[colunas] = min_max_scaler.fit_transform(X_train[colunas])
X_test[colunas] = min_max_scaler.transform(X_test[colunas])

X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)
