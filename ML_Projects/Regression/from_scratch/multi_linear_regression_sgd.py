import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
    return data.sample(frac=1).reset_index(drop=True)

def train_test_split(data, test_size=0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    split_point = int(len(data) * (1 - test_size))
    train_set = data[:split_point]
    test_set = data[split_point:]
    return train_set, test_set

def prediction(x, w, b):
    return np.dot(x, w) + b

def cost_value(y, w, x, b, m):
    pred_y = np.dot(x, w) + b
    squared_err = np.sum((pred_y - y) ** 2)
    return 1 / (2 * m) * squared_err

def stochastic_gradient_descent(w, b, x, y, m, alpha, num_iter):
    cost_history = np.zeros(num_iter)
    for i in range(num_iter):
        for j in range(m):
            x_i = np.array(x[j], dtype=np.float64)
            y_i = y[j]
            pred_y = np.dot(w, x_i) + b
            d_dw = (1 / m) * (pred_y - y_i) * x_i
            d_db = (1 / m) * (pred_y - y_i)
            w = w - (alpha * d_dw)
            b = b - (alpha * d_db)
        cost_history[i] = cost_value(y, w, x, b, m)
    return w, b, cost_history

def run_stochastic_linear_regression(test_size=0.2, alpha=0.01, num_iter=1000, plot=True):
    data = pd.read_csv('datasets/Student_Performance.csv')
    data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    shuffled_data = shuffle_data(data)
    train_data, test_data = train_test_split(shuffled_data, test_size=test_size)

    train_features = train_data[['Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    train_y = train_data['Performance Index']

    mean = train_features.mean()
    std = train_features.std()
    train_features = (train_features - mean) / std

    w_init = np.zeros(train_features.shape[1])
    b_init = 0
    m = train_y.count()

    w, b, cost_history = stochastic_gradient_descent(w_init, b_init, train_features.values, train_y.values, m, alpha, num_iter)
    predict_train = prediction(train_features.values, w, b)

    if plot:
        plt.plot(range(num_iter), cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function using Stochastic Gradient Descent')
        plt.show()

    mse_train = (1 / m) * np.sum((predict_train - train_y) ** 2)
    y_mean = np.mean(train_y)
    ss_tot = np.sum((train_y - y_mean) ** 2)
    ss_res = np.sum((train_y - predict_train) ** 2)
    r2_train = 1 - (ss_res / ss_tot)

    test_features = test_data[['Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    test_y = test_data['Performance Index']
    test_features = (test_features - mean) / std

    predict_test = prediction(test_features.values, w, b)
    m_test = test_y.count()
    mse_test = (1 / m_test) * np.sum((predict_test - test_y) ** 2)
    y_test_mean = np.mean(test_y)
    ss_tot_test = np.sum((test_y - y_test_mean) ** 2)
    ss_res_test = np.sum((test_y - predict_test) ** 2)
    r2_test = 1 - (ss_res_test / ss_tot_test)

    return {
        'weights': w,
        'bias': b,
        'cost_history': cost_history,
        'mse_train': mse_train,
        'r2_train': r2_train,
        'mse_test': mse_test,
        'r2_test': r2_test,
        'predict_train': predict_train,
        'predict_test': predict_test,
        'train_features': train_features,
        'train_y': train_y,
        'test_features': test_features,
        'test_y': test_y
    }

if __name__ == "__main__":
    results = run_stochastic_linear_regression()
    print("Final weights:", results['weights'])
    print("Final bias:", results['bias'])
    print("Train MSE:", results['mse_train'])
    print("Train R2:", results['r2_train'])
    print("Test MSE:", results['mse_test'])
    print("Test R2:", results['r2_test'])
