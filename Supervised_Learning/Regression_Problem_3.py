import numpy as np
import matplotlib.pyplot as plt
x = np.array([2, 4, 8, 7, 5])
y_actual = np.array([1, 5, 4, 12, 16])
w = 0
b = 0
num_iter = 1000
alpha = 0.0001

# Gradient Descent Algorithm
for _ in range(num_iter):
    y_pred = w * x + b
    d_w = (1/len(x)) * np.sum((y_pred - y_actual) * x)
    d_b = (1/len(x)) * np.sum(y_pred - y_actual)
    w = w - alpha * d_w
    b = b - alpha * d_b

plt.title("Univariate Linear Regression")
plt.scatter(x, y_actual, color='blue')
plt.plot(x, w * x + b, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(f"Slope (w): {w}")
print(f"Intercept (b): {b}")
