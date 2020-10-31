# Decision Tree Regression Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# We need to keep independent variable in a Matrix format, not vector format
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))"""

# Fitting Decision Tree Regression to dataset
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(random_state = 0).fit(X, y)

# Visiualizing SVR regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'green')
plt.plot(X_grid, regressor.predict(X_grid), color = 'purple')
plt.title('Truth or Bluff ( SVR Model )')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting new results
y_pred = regressor.predict([[6.5]])

