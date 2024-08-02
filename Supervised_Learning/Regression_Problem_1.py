import matplotlib.pyplot as plt

def Mean_X(x):
    total = 0
    for i in x:
        total += i
    mean = total / len(x)
    return mean

def Mean_Y(y):
    total = 0
    for i in y:
        total += i
    mean = total / len(y)
    return mean

def slope(mean_x, mean_y, x, y):
    numerator = 0
    denominator = 0
    for i, j in zip(x, y):
        numerator += (i - mean_x) * (j - mean_y)
        denominator += (i - mean_x) ** 2
    return numerator / denominator

def intercept(mean_y, slope, mean_x):
    return mean_y - (slope * mean_x)

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

mean_x = Mean_X(x)
mean_y = Mean_Y(y)
slope_m = slope(mean_x, mean_y, x, y)
intercept_b = intercept(mean_y, slope_m, mean_x)

equation = f"{slope_m}x + {intercept_b}"

print("Mean of X: ", mean_x)
print("Mean of Y: ", mean_y)
print("Slope: ", slope_m)
print("Intercept: ", intercept_b)
print("Equation: ", equation)

plt.scatter(x, y, color='blue', label='Data Points')
predicted_y = [slope_m * i + intercept_b for i in x]
plt.plot(x, predicted_y, color='red', label='Regression Line')
plt.xlabel("Independent Variable (Hours Studied)")
plt.ylabel("Dependent Variable (Test Score)")
plt.title("Hours Studied vs Test Score Regression")
plt.legend()
plt.show()
