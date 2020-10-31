import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#print(x)
#print(y)

#np.set_printoptions(threshold = np.nan)

#Handling the missing data

from sklearn.impute import SimpleImputer
imputer_obj = SimpleImputer()
imputer_obj = imputer_obj.fit(x[:, 1:3])
x[:, 1:3] = imputer_obj.transform(x[:, 1:3])