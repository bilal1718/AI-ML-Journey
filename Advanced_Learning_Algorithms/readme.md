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

## Deciding What to Try Next: Bias and Variance

When training machine learning algorithms, understanding whether your model suffers from high bias or high variance is crucial for making informed decisions on how to improve performance. Here’s a breakdown of common strategies to address these issues:

If your model makes unacceptably large errors, you might consider the following steps:

1. **Get More Training Examples**: Helps with high variance but may not solve high bias issues.
2. **Try a Smaller Set of Features**: Useful for high variance; reduces the model’s flexibility.
3. **Get Additional Features**: Helps with high bias; provides more information for the model.
4. **Add Polynomial Features**: Addresses high bias by allowing the model to fit more complex patterns.
5. **Decrease Lambda**: Fixes high bias by reducing regularization and increasing model complexity.
6. **Increase Lambda**: Addresses high variance by reducing model complexity and overfitting.

### Additional Notes

- **Reducing Training Set Size**: This strategy does not help with high bias and is generally not recommended. Reducing the training set may improve training error but usually worsens cross-validation error.
- **Mastery of Bias and Variance**: Understanding bias and variance is fundamental, but mastering these concepts requires ongoing practice and experience.




## Neural Networks and the Bias-Variance Tradeoff

### Neural Networks and Bias

Neural networks, particularly large ones, have the capability to model very complex functions. This allows for a good fit on the training data, which can help in reducing bias.

- **High Bias**: If a neural network is underperforming on the training set (high bias), consider increasing the network size by adding more layers or more units per layer to reduce bias.

### Neural Networks and Variance

After addressing high bias by increasing the network size, you may encounter high variance (overfitting), where the network performs well on the training data but poorly on validation or test data.

To tackle high variance, you can:

1. **Get More Data**: Increasing the amount of training data helps the model generalize better and reduces overfitting.
2. **Regularize the Model**: Use techniques such as L2 regularization (weight decay) to add a penalty to the loss function based on the magnitude of the weights, which helps in preventing overfitting.

### Computational Considerations

- **Resource Requirements**: Larger neural networks require more computational resources and time to train. Modern hardware, especially GPUs, can accelerate this process, but very large networks may still be resource-intensive.

### Regularization in Neural Networks

- **Purpose**: Regularization helps to prevent overfitting by penalizing large weights.
- **Implementation**: In frameworks like TensorFlow, L2 regularization can be added by including a regularization term in the loss function.

### Practical Advice for Neural Network Training

- **Use Larger Networks**: Generally, using a larger network is beneficial, provided you implement appropriate regularization techniques and have sufficient computational resources.
- **Monitor Bias and Variance**: Track both bias (training error) and variance (cross-validation error) to guide adjustments. Increase the model size if bias is high. If variance is high, consider acquiring more data or applying regularization.