#                 Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorial data

# Encoding independent variable - due to 'State' column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])
one_hot_encoder = OneHotEncoder()
column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(column_transformer.fit_transform(X), dtype = float)

# Avoiding Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# X_train = X_train.astype(np.float64)
# X_test = X_test.astype(np.float64)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Multiple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train, y_train)

# Predicting the test set results
y_predict = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
# X = np.append(arr = np.ones((50, 1)), values = X, axis = 1)
X = sm.add_constant(X) 

# Removing the insignificant independent variables   

X_optimum = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_optimum).fit()
regressor_OLS.summary()

X_optimum = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_optimum).fit()
regressor_OLS.summary()

X_optimum = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_optimum).fit()
regressor_OLS.summary()

X_optimum = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(y, X_optimum).fit()
regressor_OLS.summary()

X_optimum = X[:, [0, 3]]
regressor_OLS = sm.OLS(y, X_optimum).fit()
regressor_OLS.summary()

