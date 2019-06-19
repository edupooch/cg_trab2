import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

dataset = pd.read_csv('features.csv')
y = dataset.pop('class')
X = dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train['class'] = y_train
X_train.to_csv('train.csv', index=False)

X_test['class'] = y_test
X_test.to_csv('test.csv', index=False)
