import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
X = np.array([[2, 1],[3,2],[3,4],[5,4]]) 
y = np.array([[1],[0],[1],[0]])
alpha = 0.01
num_iters = 1000

input_layer_size = X.shape[1]
hidden_layer_size = 2
output_layer_size = 1

W1 = np.random.randn(input_layer_size, hidden_layer_size)
b1 = np.zeros((1, hidden_layer_size))
W2 = np.random.randn(hidden_layer_size, output_layer_size)
b2 = np.zeros((1, output_layer_size))

for _ in range(num_iters):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    error = a2 - y

    W2 -= alpha * np.dot(a1.T, error) / X.shape[0]
    b2 -= alpha * np.sum(error, axis=0, keepdims=True) / X.shape[0]
    
    a1_error = np.dot(error, W2.T) * sigmoid_derivative(z1)
    W1 -= alpha * np.dot(X.T, a1_error) / X.shape[0]
    b1 -= alpha * np.sum(a1_error, axis=0, keepdims=True) / X.shape[0]

def neural_network(X, W1, b1, W2, b2):
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden_layer, W2) + b2)
    return output

predictions = neural_network(X, W1, b1, W2, b2)
print(predictions)
binary_predictions = (predictions >= 0.5).astype(int)

correct_predictions = np.sum(binary_predictions == y)
total_predictions = y.size
accuracy = correct_predictions / total_predictions

print(f'Accuracy: {accuracy * 100:.2f}%')
