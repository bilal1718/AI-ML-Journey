# Best Practices for Training ML Model

## Evaluating The Model

* Split the data into three parts:
  - **Training Set**
  - **Cross Validation Set**
  - **Test Set**

* Train the model using the training set, then perform cross-validation to find the best model based on the lowest loss value.
* Finally, test the selected model on the test set to evaluate its performance.

## Diagnosing the Bias and Variance

### Key Concepts

1. **Bias**
   - **Definition**: The error introduced by approximating a real-world problem, which may be complex, by a simplified model.
   - **High Bias**: Indicates underfitting, where the model is too simple to capture the underlying trend in the data. This results in poor performance on both training and validation sets.

2. **Variance**
   - **Definition**: The error introduced by the model's sensitivity to fluctuations in the training dataset.
   - **High Variance**: Indicates overfitting, where the model is too complex and captures noise in the training data rather than the underlying trend. This results in good performance on training data but poor performance on validation data.

3. **Training Error**
   - **Definition**: The error of the model on the training dataset.
   - **Significance**: Indicates how well the model fits the training data.

4. **Cross-Validation Error**
   - **Definition**: The error of the model on a separate cross-validation dataset that was not used during training.
   - **Significance**: Provides insight into how well the model generalizes to unseen data.

### Diagnosing Bias and Variance

1. **High Bias (Underfitting)**
   - **Indicators**: High training error and high cross-validation error.
   - **Actions**:
     - Use a more complex model.
     - Add more features or polynomial terms.
     - Consider reducing regularization.

2. **High Variance (Overfitting)**
   - **Indicators**: Low training error but high cross-validation error.
   - **Actions**:
     - Use simpler models.
     - Reduce the complexity of the model (e.g., reduce polynomial degree).
     - Apply regularization techniques (e.g., L1 or L2 regularization).
     - Gather more training data.

3. **Balanced Model**
   - **Indicators**: Both training error and cross-validation error are low and similar.
   - **Actions**: Continue with the current model or fine-tune hyperparameters for potential improvements.

### Practical Example

Below is a Python code snippet to demonstrate diagnosing bias and variance using a simple linear regression model. This example calculates training and cross-validation errors:

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.flatten() + np.random.normal(0, 1, X.shape[0])

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def model(X, theta):
    return X @ theta

def cost_function(X, y, theta):
    m = len(y)
    predictions = model(X, theta)
    return np.sum((predictions - y) ** 2) / (2 * m)

X_train_poly = np.hstack([X_train, X_train ** 2])
X_test_poly = np.hstack([X_test, X_test ** 2])
theta_best = np.linalg.inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train

J_train = cost_function(X_train_poly, y_train, theta_best)
J_cv = cost_function(X_test_poly, y_test, theta_best)

print(f"Training Error (Mean Squared Error): {J_train:.2f}")
print(f"Cross-Validation Error (Mean Squared Error): {J_cv:.2f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model(X_train_poly, theta_best), color='red', label='Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data and Fit')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_test, model(X_test_poly, theta_best), color='red', label='Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Test Data and Fit')
plt.legend()

plt.tight_layout()
plt.show()
