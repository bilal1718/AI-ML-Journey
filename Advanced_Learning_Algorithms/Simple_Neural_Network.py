import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[2, 1]]) 
y = np.array([[1]]) 
alpha = 0.01
num_iters = 1000
m = X.shape[0]

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
    d_a2 = error * a2 * (1 - a2)
    d_z2 = d_a2
    d_W2 = np.dot(a1.T, d_z2) / m
    d_b2 = np.sum(d_z2, axis=0, keepdims=True) / m
    
    d_a1 = np.dot(d_z2, W2.T) * a1 * (1 - a1)
    d_W1 = np.dot(X.T, d_a1) / m
    d_b1 = np.sum(d_a1, axis=0, keepdims=True) / m
    
    W1 -= alpha * d_W1
    b1 -= alpha * d_b1
    W2 -= alpha * d_W2
    b2 -= alpha * d_b2

def neural_network(X, W1, b1, W2, b2):
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden_layer, W2) + b2)
    return output

prediction = neural_network(X, W1, b1, W2, b2)
print(prediction)
