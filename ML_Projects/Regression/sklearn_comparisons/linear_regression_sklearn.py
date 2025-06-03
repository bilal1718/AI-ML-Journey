import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



# Univariate Linear Regression
data=pd.read_csv('datasets/Salary_dataset.csv')
data.drop(columns=['Unnamed: 0'], inplace=True)

x = data["YearsExperience"].values.reshape(-1, 1)
y=data["Salary"]

reg = LinearRegression().fit(x, y)
print("Mean Squared Error:", mean_squared_error(y, reg.predict(x)))
print("Model Score (R^2):", reg.score(x, y))
print("Coefficient (slope):", reg.coef_)
print("Intercept:", reg.intercept_)

#Multi Linear Regression
data = pd.read_csv('datasets/Student_Performance.csv')
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

shuffled_data = shuffle(data,random_state=0)
x = shuffled_data[['Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y=shuffled_data['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

reg = LinearRegression().fit(X_train, y_train)
print("Below is Multi Linear for training data ")
print("Mean Squared Error:", mean_squared_error(y_train, reg.predict(X_train)))
print("Model Score (R^2):", reg.score(X_train, y_train))
print("Coefficient (slope):", reg.coef_)
print("Intercept:", reg.intercept_)

print("Below is Multi Linear for test data ")

print("Mean Squared Error:", mean_squared_error(y_test, reg.predict(X_test)))
print("Model Score (R^2):", reg.score(X_test, y_test))
print("Coefficient (slope):", reg.coef_)
print("Intercept:", reg.intercept_)