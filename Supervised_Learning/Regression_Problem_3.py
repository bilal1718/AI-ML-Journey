import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures


x = np.array([2, 4, 8, 7, 5])
y_actual = np.array([1, 5, 4, 12, 16])
x_reshaped = x.reshape(-1, 1)  

# below is simple linear regression using sklearn which gives same result as the custom implementation below.
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_reshaped)

reg = LR()
reg.fit(x_poly, y_actual)

sklearn_w = reg.coef_[0]
sklearn_b = reg.intercept_

print(f"Sklearn Slope (w): {sklearn_w}")
print(f"Sklearn Intercept (b): {sklearn_b}")


# Custom implementation of univariate linear regression
w = 0
b = 0
num_iter = 10000
alpha = 0.01

def w_b_update(w, b, x, y_actual, alpha):
    for _ in range(num_iter):
        y_pred = w * x + b
        d_w = (1/len(x)) * np.sum((y_pred - y_actual) * x)
        d_b = (1/len(x)) * np.sum(y_pred - y_actual)
        w -= alpha * d_w
        b -= alpha * d_b
    return w, b

w, b = w_b_update(w, b, x, y_actual, alpha)
print(f"Custom Slope (w): {w}")
print(f"Custom Intercept (b): {b}")

predicted_custom = w * x + b
predicted_sklearn = sklearn_w * x + sklearn_b

# Plotting the results
plt.title("Univariate Linear Regression")
plt.scatter(x, y_actual, color='blue', label='Actual Data')
plt.plot(x, predicted_custom, color='red', label='Custom Model')
plt.plot(x, predicted_sklearn, color='green', linestyle='--', label='Sklearn Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()





# See how sklearn fits data with using polynomial features to degree 4.
x = np.array([2, 4, 8, 7, 5]).reshape(-1, 1)
y = np.array([1, 5, 4, 12, 16])

poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)

model = LR()
model.fit(x_poly, y)

y_pred = model.predict(x_poly)

plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='green')
plt.title("Polynomial Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()