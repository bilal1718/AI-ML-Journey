import numpy as np
import matplotlib.pyplot as plt

age = np.array([3, 5, 2, 8, 7, 6, 1, 9, 4, 5])
mileage = np.array([30000, 50000, 20000, 80000, 70000, 60000, 10000, 90000, 40000, 50000])
horsepower = np.array([150, 130, 200, 100, 180, 170, 220, 90, 160, 140])
num_doors = np.array([4, 2, 4, 4, 2, 4, 2, 4, 4, 2])

price = np.array([20000, 15000, 25000, 12000, 18000, 16000, 27000, 11000, 22000, 17000])

X = np.column_stack((age, mileage, horsepower, num_doors))

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std
print(X_scaled)
m = X_scaled.shape[0]

w = np.zeros(X_scaled.shape[1])
b = 0
alpha = 0.001
iterations = 10000

for _ in range(iterations):
    y_pred = np.dot(X_scaled, w) + b
    cost = y_pred - price
    gradient_w = (1 / m) * np.dot(X_scaled.T, cost)
    gradient_b = (1 / m) * np.sum(cost)

    w -= alpha * gradient_w
    b -= alpha * gradient_b

print("Weights:", w)
print("Bias:", b)

y_pred = np.dot(X_scaled, w) + b

features = ['Age', 'Mileage', 'Horsepower', 'Num_Doors']
for i in range(X_scaled.shape[1]):
    plt.figure()
    plt.scatter(X[:, i], price, color='blue', label='Actual Price')
    plt.scatter(X[:, i], (X_scaled[:, i] * w[i] + b), color='red', label='Predicted Price')
    plt.xlabel(features[i])
    plt.ylabel('Price')
    plt.legend()
    plt.title(f'{features[i]} vs Price')
    plt.show()
