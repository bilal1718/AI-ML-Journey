import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def weights(x, point, tau):
    m = x.shape[0]
    W = np.eye(m)
    for i in range(m):
        diff = x[i] - point
        W[i, i] = np.exp(-(diff**2) / (2 * tau**2))
    return W

data = pd.read_csv('datasets/Salary_dataset.csv')
data.drop(columns=['Unnamed: 0'], inplace=True)

x = data["YearsExperience"].values
y = data["Salary"].values

X = np.vstack((np.ones(len(x)), x)).T

def lwlr_predict(point, X, y, tau):
    W = weights(x, point, tau)
    XTWX = X.T @ W @ X
    try:
        theta = np.linalg.inv(XTWX) @ X.T @ W @ y
        print("Theta with inverse: ", theta)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(XTWX) @ X.T @ W @ y
        print("Theta with psuedo inverse: ", theta)

    point_vec = np.array([1, point])
    prediction = point_vec @ theta
    return prediction

point = 3.4
tau = 0.7
prediction = lwlr_predict(point, X, y, tau)
print(f"Predicted salary at {point} years experience: {prediction}")

