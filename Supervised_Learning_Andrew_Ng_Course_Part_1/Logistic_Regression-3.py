import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [25, 50000, 1, 10, 0],
    [30, 60000, 0, 15, 1],
    [35, 70000, 1, 20, 0],
    [40, 80000, 0, 25, 1],
    [45, 90000, 1, 30, 0],
    [50, 100000, 0, 35, 1]
])

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

Y = np.array([0, 1, 1, 1, 0, 1])
m = X.shape[0]
b = 0
w = np.zeros(X.shape[1])
alpha = 0.001
num_iters = 1000

def sigmoid(pred):
    return 1 / (1 + np.exp(-np.clip(pred, -500, 500)))

def gradient_descent():
    global w, b
    d_db = 0
    d_dw = np.zeros(X.shape[1])
    
    for i in range(m):
        f_x = sigmoid(np.dot(X[i], w) + b)
        error = f_x - Y[i]
        
        d_db += error
        d_dw += error * X[i]

    d_db /= m
    d_dw /= m
    
    return d_dw, d_db

def logistic_regression():
    global w, b
    for _ in range(num_iters):
        df_dw, df_db = gradient_descent()

        w -= alpha * df_dw
        b -= alpha * df_db

    return w, b

w_pred, b_pred = logistic_regression()

def plot_decision_boundary(X, Y, w, b):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='blue', label='Class 1')
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], color='red', label='Class 0')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], w_pred[:2]) + b_pred)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Logistic Regression Decision Boundary')
    plt.show()
plot_decision_boundary(X, Y, w_pred, b_pred)
pred = sigmoid(np.dot(X, w_pred) + b_pred)
threshold = 0.5
binary_preds = (pred >= threshold).astype(int)
correct_predictions = np.sum(binary_preds == Y)
accuracy = correct_predictions / m

print("The predictions are: ", binary_preds)
print("Accuracy: ", accuracy)
