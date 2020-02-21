from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv(Path('./winequality-red.csv'))

# Working with Numeric Features
n_features = train.select_dtypes(include=[np.number])
corr = n_features.corr()
print(corr['quality'].sort_values(ascending=False)[:3], '\n')

# #handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print('Handling missing values:')
print(sum(data.isnull().sum() != 0))

# Build a linear model
y = np.log(data.quality)
X = data.drop(['quality'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the performance and visualize results
print("R^2 is: ", model.score(X_test, y_test))
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print('RMSE is: ', mean_squared_error(y_test, predictions))