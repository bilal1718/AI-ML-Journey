import numpy as np
import matplotlib.pyplot as plt
age = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
actual_blood_pressure = np.array([120, 122, 125, 130, 135, 140, 145, 150, 155, 160])
def model_equation(m, age, b):
    return m * age + b

m = 0.9284848484848545
b = 94.09696969696978
predicted_bp = model_equation(m, age, b)
plt.scatter(age, actual_blood_pressure, label='Actual Data')
plt.plot(age, predicted_bp, color='red', label='Linear Regression Line')
plt.title("Blood Pressure vs Age")
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.legend()
plt.grid(True)
plt.show()
