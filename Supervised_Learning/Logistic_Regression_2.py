import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def predict_probabilities(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
def update_parameters(X, y_true, y_pred, weights, bias, learning_rate):
    m = X.shape[0]
    dw = np.dot(X.T, (y_pred - y_true)) / m
    db = np.sum(y_pred - y_true) / m
    weights -= learning_rate * dw
    bias -= learning_rate * db
    return weights, bias
def train_logistic_regression(X_train, y_train, epochs, learning_rate):
    n_features = X_train.shape[1]
    weights = np.zeros((n_features, 1))
    bias = 0

    for epoch in range(epochs):
        y_pred = predict_probabilities(X_train, weights, bias)
        loss = compute_loss(y_train, y_pred)
        weights, bias = update_parameters(X_train, y_train, y_pred, weights, bias, learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return weights, bias
def predict(X, weights, bias):
    y_pred_prob = predict_probabilities(X, weights, bias)
    return (y_pred_prob >= 0.5).astype(int)
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
passes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(-1, 1)
X_train, X_test = hours[:7], hours[7:]
y_train, y_test = passes[:7], passes[7:]
epochs = 1000
learning_rate = 0.01

weights, bias = train_logistic_regression(X_train, y_train, epochs, learning_rate)

y_pred = predict(X_test, weights, bias)
print("Predictions:", y_pred.flatten())
print("Actual Values:", y_test.flatten())
print("Accuracy:", accuracy(y_test, y_pred))