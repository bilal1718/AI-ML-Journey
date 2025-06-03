import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
    return data.sample(frac=1).reset_index(drop=True)

def train_test_split(data, test_size):
    split_point = int(len(data) * (1 - test_size))
    train_data = data[:split_point]
    test_data = data[split_point:]
    return train_data, test_data

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def cost_function(x, y, w, b, m):
    z = np.dot(x, w) + b
    h = sigmoid(z)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    return (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))

def gradient_descent(x, y, w, b, m, num_iter, alpha):
    cost_history = np.zeros(num_iter)
    for i in range(num_iter):
        z = np.dot(x, w) + b
        h = sigmoid(z)
        d_dw = (1 / m) * (x.T @ (h - y))
        d_db = (1 / m) * np.sum(h - y)
        w = w - alpha * d_dw
        b = b - alpha * d_db
        cost_history[i] = cost_function(x, y, w, b, m)
    return w, b, cost_history

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Rain'] = data['Rain'].map({'rain': 1, 'no rain': 0})
    data = shuffle_data(data)
    train_data, test_data = train_test_split(data, 0.2)
    return train_data, test_data

def normalize_features(train_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    train_x_norm = (train_x - mean) / std
    test_x_norm = (test_x - mean) / std
    return train_x_norm, test_x_norm

def prepare_data(train_data, test_data):
    features = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
    train_x = train_data[features]
    train_y = train_data['Rain']
    test_x = test_data[features]
    test_y = test_data['Rain']

    train_x_norm, test_x_norm = normalize_features(train_x, test_x)

    return train_x_norm.values, train_y.values, test_x_norm.values, test_y.values

def train_logistic_regression(train_x, train_y, num_iter=1000, alpha=0.01):
    m, n = train_x.shape
    w = np.zeros(n)
    b = 0
    w, b, cost_history = gradient_descent(train_x, train_y, w, b, m, num_iter, alpha)
    return w, b, cost_history

def predict(x, w, b, threshold=0.5):
    prob = sigmoid(np.dot(x, w) + b)
    return (prob >= threshold).astype(int), prob

def evaluate_model(preds, true):
    epsilon = 1e-15
    TP = np.sum((preds == 1) & (true == 1))
    TN = np.sum((preds == 0) & (true == 0))
    FP = np.sum((preds == 1) & (true == 0))
    FN = np.sum((preds == 0) & (true == 1))

    accuracy = np.mean(preds == true)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return accuracy, precision, recall, f1_score

def plot_cost(cost_history):
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.grid(True)
    plt.show()

def print_data_summary(train_y, test_y):
    print('Rain Days in training set: ', train_y.sum())
    print('No Rain Days in training set: ', len(train_y) - train_y.sum())
    print('Rain Days in test set: ', test_y.sum())
    print('No Rain Days in test set: ', len(test_y) - test_y.sum())

if __name__ == "__main__":
    train_data, test_data = load_and_preprocess_data('datasets/weather_forecast_data.csv')
    train_x, train_y, test_x, test_y = prepare_data(train_data, test_data)
    
    print_data_summary(train_y, test_y)
    
    w, b, cost_history = train_logistic_regression(train_x, train_y, num_iter=1000, alpha=0.01)
    plot_cost(cost_history)
    
    train_preds, _ = predict(train_x, w, b, threshold=0.3)
    accuracy, precision, recall, f1_score = evaluate_model(train_preds, train_y)
    print("Training set metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    
    test_preds, _ = predict(test_x, w, b, threshold=0.35)
    accuracy, precision, recall, f1_score = evaluate_model(test_preds, test_y)
    print("\nTest set metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")


