import numpy as np

def fit_decision_tree(X, y):
    def split_data(X, y, feature_index, value):
        mask = X[:, feature_index] == value
        return X[mask], y[mask]

    if len(set(y)) == 1:
        return y[0]
    
    if np.all(X[:, 0] == 0):
        return 0
    elif np.all(X[:, 0] == 1):
        return 1
    else:
        return 0 if np.mean(y) < 0.5 else 1

def predict(tree, X):
    return np.array([tree for _ in range(X.shape[0])])

X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
y = np.array([0, 1, 1, 0])

tree = fit_decision_tree(X, y)
print("Decision Tree Prediction Logic: ", tree)

X_test = np.array([[1, 1], [0, 0]])
predictions = predict(tree, X_test)
print("Predictions for test data:", predictions)
