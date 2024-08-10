# Best Practices for Training ML Models

## Evaluating The Model

* **Split the data into three parts:**
  - **Training Set**
  - **Cross Validation Set**
  - **Test Set**

* **Train the model** using the training set, then perform cross-validation to find the best model based on the lowest loss value.
* **Finally, test the selected model** on the test set to evaluate its performance.

## Diagnosing Bias and Variance

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

## Establishing a Baseline Level of Performance

When evaluating your model, it's crucial to establish a baseline level of performance to accurately diagnose bias and variance.

**What is a Baseline Performance?**
- **Baseline Performance** is a reference point that helps you gauge how well your learning algorithm is performing relative to a reasonable benchmark. It helps to understand if the performance of your model is good enough or if there's room for improvement.

**Why Establish a Baseline?**
- **Human-Level Performance**: For tasks like speech recognition, where even humans make mistakes (e.g., 10.6% error), comparing your model's performance to human performance can be a useful benchmark. This helps you understand whether your model's performance is near the best possible given the task's inherent challenges.
- **Competing Algorithms**: If there are existing benchmarks or previous implementations, comparing against these can also set a practical baseline.

**How to Use the Baseline:**
- **High Bias**: If your model's error is much worse than the baseline, you have a high bias problem. The model is underfitting.
- **High Variance**: If there is a large gap between training and cross-validation errors, the model is overfitting.

**Example Metrics to Watch:**
- **Difference Between Training Error and Baseline**: Indicates bias. A large difference suggests high bias.
- **Difference Between Training Error and Cross-Validation Error**: Indicates variance. A large difference suggests high variance.

## Learning Curves

**Learning Curves** are graphical representations that show how the model's performance changes with different sizes of training data. They are useful for diagnosing bias and variance in your model.

### What Are Learning Curves?

- **Training Learning Curve**: Shows how the training error decreases as the size of the training data increases.
- **Validation Learning Curve**: Shows how the validation error changes with increasing training data.

### Why Use Learning Curves?

1. **Diagnose Model Performance**: Learning curves help visualize how well the model is learning. If the training error is high and the validation error is also high, the model might be underfitting. If the training error is low but the validation error is high, the model might be overfitting.
2. **Determine if More Data is Needed**: If both errors are still high, you might need more training data. If the validation error starts to level off while the training error continues to decrease, you might be capturing noise with more data.

### How to Interpret Learning Curves

1. **High Bias (Underfitting)**
   - **Indicators**: Both training and validation errors are high and converge to a high value.
   - **Actions**: Increase the complexity of the model or add more features.

2. **High Variance (Overfitting)**
   - **Indicators**: Training error is low but validation error is high and does not decrease as training data increases.
   - **Actions**: Simplify the model or apply regularization techniques.

### Practical Example

Below is a Python code snippet to generate learning curves for a simple linear regression model:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.flatten() + np.random.normal(0, 1, X.shape[0])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

train_errors, val_errors = [], []

for m in range(1, len(X_train)):
    model = LinearRegression()
    model.fit(X_train[:m], y_train[:m])
    
    y_train_predict = model.predict(X_train[:m])
    y_val_predict = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
    val_errors.append(mean_squared_error(y_test, y_val_predict))

plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, len(X_train)), train_errors, label='Training Error')
plt.plot(np.arange(1, len(X_train)), val_errors, label='Validation Error')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves')
plt.legend()
plt.show()
