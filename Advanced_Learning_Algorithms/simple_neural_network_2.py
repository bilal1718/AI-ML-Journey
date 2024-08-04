import numpy as np

x = np.array([[23, 34, 54, 65, 32],
              [45, 67, 89, 21, 32],
              [90, 80, 70, 60, 50],
              [34, 67, 62, 76, 44],
              [89, 79, 67, 45, 32]])
y = np.array([[1, 0], 
              [0, 1],
              [1, 0],
              [0, 1],
              [1, 0]])

np.random.seed(0)

w1 = np.random.randn(5, 4) * 0.01  
b1 = np.random.randn(4) * 0.01

w2 = np.random.randn(4, 3) * 0.01
b2 = np.random.randn(3) * 0.01

w3 = np.random.randn(3, 2) * 0.01
b3 = np.random.randn(2) * 0.01

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense_layer(A_in, W, b, activation=sigmoid):
    z = np.matmul(A_in, W) + b
    f_out = activation(z)
    return f_out

def sequential(X):
    a1 = dense_layer(X, w1, b1, activation=sigmoid)
    a2 = dense_layer(a1, w2, b2, activation=sigmoid)
    a3 = dense_layer(a2, w3, b3, activation=sigmoid) 
    return a3

output = sequential(x)
print(output)
