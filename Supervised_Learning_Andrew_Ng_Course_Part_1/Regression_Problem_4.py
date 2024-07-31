import numpy as np
import matplotlib.pyplot as plt

def linear_regression_fit(X, y, alpha=0.001, num_iters=1000):
    w = 0
    b = 0
    m = len(X)

    for _ in range(num_iters):
        y_pred = w * X + b

        d_w = (1/m) * np.sum((y_pred - y) * X)
        d_b = (1/m) * np.sum(y_pred - y)

        w =w - alpha * d_w
        b =b - alpha * d_b

    return w, b

def predict_sales(temp, w, b):
    return w * temp + b

temp = np.array([14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
ice_cream_sales_actual = np.array([215, 325, 185, 332, 406, 522, 412, 614, 544, 421])

w_opt, b_opt = linear_regression_fit(temp, ice_cream_sales_actual)

try:
    temp_input = float(input("Enter the temperature to predict sales: "))
except ValueError:
    print("Invalid input. Please enter a valid number.")
    exit()

sales_prediction = predict_sales(temp_input, w_opt, b_opt)
print(f"Predicted sales for temperature {temp_input}°C: {sales_prediction:.2f} $")
plt.title("Temperature vs. Ice Cream Sales")
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales ($)")
plt.scatter(temp, ice_cream_sales_actual, label="Actual Sales")
plt.plot(temp, w_opt * temp + b_opt, color='red', label="Predicted Sales")
plt.scatter(temp_input, sales_prediction, color='green', label="Predicted Value")
plt.legend()
plt.show()
