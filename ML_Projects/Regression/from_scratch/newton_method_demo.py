import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('datasets/weather_forecast_data.csv')
data['Rain'] = data['Rain'].map({'rain': 1, 'no rain': 0})

x = data[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y = data['Rain']

x = (x - x.mean(axis=0)) / x.std(axis=0)
x = np.c_[np.ones(x.shape[0]), x]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(h, y):
    epsilon = 1e-15
    return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

def newton_raphson(X, y, iterations=10):
    m, n = X.shape
    theta = np.zeros(n)
    costs = []

    for _ in range(iterations):
        h = sigmoid(X @ theta)
        cost = compute_cost(h, y)
        costs.append(cost)

        gradient = X.T @ (h - y)
        R = np.diag(h * (1 - h))
        H = X.T @ R @ X
        theta -= np.linalg.inv(H) @ gradient

    return theta, costs

theta, cost_history = newton_raphson(x, y)

pred_probs = sigmoid(x @ theta.T)
threshold = 0.3
class_labels = (pred_probs >= threshold).astype(int)

accuracy = np.mean(y == class_labels)
print("Accuracy:", accuracy)

TP = np.sum((class_labels == 1) & (y == 1))
TN = np.sum((class_labels == 0) & (y == 0))
FP = np.sum((class_labels == 1) & (y == 0))
FN = np.sum((class_labels == 0) & (y == 1))
epsilon = 1e-15
precision = TP / (TP + FP + epsilon)
recall = TP / (TP + FN + epsilon)
f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cost_history)+1), cost_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Cost (Log Loss)")
plt.title("Cost vs Iteration (Newton-Raphson)")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(pred_probs[y==0], bins=20, alpha=0.6, label='No Rain (0)', color='skyblue')
plt.hist(pred_probs[y==1], bins=20, alpha=0.6, label='Rain (1)', color='salmon')
plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold}')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, class_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import precision_recall_curve

prec, rec, _ = precision_recall_curve(y, pred_probs)

plt.figure(figsize=(8, 5))
plt.plot(rec, prec, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()
