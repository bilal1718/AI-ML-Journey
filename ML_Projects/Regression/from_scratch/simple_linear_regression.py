import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prediction(x, w, b):
    return w * x + b

def compute_cost(x, y, w, b, m):
    y_pred = w * x + b
    take_sum = np.sum((y_pred - y) ** 2)
    return (1 / (2 * m)) * take_sum

def batch_gradient_descent(x, y, w, b, m, num_iter, alpha):
    cost_history = np.ones(num_iter)
    for i in range(num_iter):
        y_pred = w * x + b
        d_dw = (1 / m) * np.sum((y_pred - y) * x)
        d_db = (1 / m) * np.sum((y_pred - y))
        w = w - (alpha * d_dw)
        b = b - (alpha * d_db)
        cost_history[i] = compute_cost(x, y, w, b, m)
    return w, b, cost_history

def run_scratch_linear_regression(plot=True):
    data = pd.read_csv('datasets/Salary_dataset.csv')
    data.drop(columns=['Unnamed: 0'], inplace=True)

    x = data["YearsExperience"]
    y = data["Salary"]

    w = 0
    b = 0
    m = x.count()
    num_iter = 1000
    alpha = 0.01

    w, b, cost_history = batch_gradient_descent(x, y, w, b, m, num_iter, alpha)
    y_pred = prediction(x, w, b)

    if plot:
        plt.plot(range(num_iter), cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function using Batch Gradient Descent')
        plt.show()

        plt.scatter(x, y, color='black')
        plt.plot(x, y_pred, color='red')
        plt.legend(['Regression Line', 'Real data'])
        plt.xlabel('Years Of experience')
        plt.ylabel('Salary')
        plt.title('Years Of experience vs Salary')
        plt.show()

    mse = (1 / m) * np.sum((y_pred - y) ** 2)

    y_mean = np.mean(y)
    SS_tot = np.sum((y - y_mean) ** 2)
    SS_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - (SS_res / SS_tot)

    return {
        'weight': w,
        'bias': b,
        'mse': mse,
        'r2': r2,
        'y_pred': y_pred,
        'x': x,
        'y': y
    }

if __name__ == "__main__":
    results = run_scratch_linear_regression()
    print("Final weight (w):", results['weight'])
    print("Final bias (b):", results['bias'])
    print("Mean Squared Error is : ", results['mse'])
    print("R² Score:", results['r2'])
