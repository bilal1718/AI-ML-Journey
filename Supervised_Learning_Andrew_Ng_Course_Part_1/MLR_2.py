import numpy as np
import matplotlib.pyplot as plt

price = np.array([10, 15, 12, 20, 8, 25, 18, 11, 9, 22])
advertise_bud = np.array([1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 900, 2200])
num_stores = np.array([20, 25, 18, 30, 15, 35, 28, 22, 16, 32])
prod_rating = np.array([4.5, 4.0, 4.2, 4.8, 3.8, 4.9, 4.6, 4.3, 3.9, 4.7])

annual_sales = np.array([15000, 20000, 17000, 30000, 10000, 35000, 25000, 16000, 11000, 32000])

X = np.column_stack((price, advertise_bud, num_stores, prod_rating))

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_scaling = (X - mean) / std

w = np.zeros(X_scaling.shape[1])
b = 0
alpha = 0.01
num_iters = 1000
m = X_scaling.shape[0]

for _ in range(num_iters):
    y_pred = np.dot(X_scaling, w) + b
    cost = y_pred - annual_sales
    gradient_w = (1 / m) * np.dot(X_scaling.T, cost)
    gradient_b = (1 / m) * np.sum(cost)
    w = w - alpha * gradient_w
    b = b - alpha * gradient_b
print("Weights:", w)
print("Bias:", b)

features = ["Price", "Advertising budget", "Number of Stores", "Product Rating"]
for i in range(len(w)):
    plt.figure()
    plt.scatter(X_scaling[:, i], annual_sales, color='blue', label='Actual Price')
    plt.scatter(X_scaling[:, i], (X_scaling[:, i] * w[i] + b), color='red', label='Predicted Price')
    plt.xlabel(features[i])
    plt.ylabel('Annual Sales')
    plt.legend()
    plt.title(f'{features[i]} vs Annual Sales')
    plt.show()
