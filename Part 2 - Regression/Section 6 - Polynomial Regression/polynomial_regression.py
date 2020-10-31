#                         Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# We need to keep independent variable in a Matrix format, not vector format
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Since the dataset is less, we need no splitting up of data

# Fitting Linear Regression model to the Dataset
from sklearn.linear_model import LinearRegression 
linear_regressor = LinearRegression().fit(X, y)

# Fitting polynomial Regression model to the Dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 5)
X_polynomial = polynomial_regressor.fit_transform(X)
polynomial_linear_regressor = LinearRegression().fit(X_polynomial, y)

# Visualizing the Linear Regression result
plt.scatter(X, y, color = 'green')
plt.plot(X, linear_regressor.predict(X), color = 'purple')
plt.title('Truth or Bluff ( Linear Model )')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression result

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'green')
plt.plot(X_grid, polynomial_linear_regressor.predict(polynomial_regressor.fit_transform(X_grid)), color = 'purple')
plt.title('Truth or Bluff ( Polynomial Model )')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Prediction Comparision
# Linear Model:
linear_regressor.predict([[6.5]])

# Polynomial Model:
polynomial_linear_regressor.predict(polynomial_regressor.fit_transform([[6.5]]))

