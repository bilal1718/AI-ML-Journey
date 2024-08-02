import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_input = np.array([600, 700, 550, 750])
y = np.array([1, 1, 1, 0])
def feature_scaling(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / (std)

X_scaled = feature_scaling(x_input)

w = 0
b = 0
alpha = 0.01
num_iters = 1000
m = x_input.shape[0]

for _ in range(num_iters):
    pred_y = np.dot(w, X_scaled) + b
    df_dw = (1 / m) * np.dot((pred_y - y), X_scaled)
    df_db = (1 / m) * np.sum(pred_y - y)
    w = w - alpha * df_dw
    b = b - alpha * df_db

z = np.dot(w, X_scaled) + b
probabilities = 1 / (1 + np.exp(-z))

pred_y = np.where(probabilities > 0.5, 1, 0)

print("The prediction will default a loan is")
print(pred_y)

accuracy = np.mean(pred_y == y)
print("Accuracy:", accuracy)

df = pd.DataFrame({'Credit Score': x_input, 'Defaulted': y})
plt.scatter(df['Credit Score'], df['Defaulted'])
plt.xlabel('Credit Score')
plt.ylabel('Defaulted')
plt.title('Credit Score vs Defaulted')
plt.show()