import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


data=pd.read_csv('Softmax_Regression/bmi.csv')
data['Gender']=data['Gender'].map({'Male':1, 'Female':0})
X=data[['Gender', 'Height','Weight']].values
X=(X - np.mean(X, axis=0))/np.std(X, axis =0)
y=data['Index'].values
num_samples, num_features = X.shape
num_classes = len(np.unique(y))

print(num_samples , num_features)
print(num_classes)
print(np.unique(y))

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

y_encoded = one_hot_encode(y, num_classes)

np.random.seed(42)
W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros((1, num_classes))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return loss

def train(X, y_encoded, W, b, learning_rate=0.1, epochs=1000):
    losses = []
    for i in range(epochs):
        z = np.dot(X, W) + b     
        y_pred = softmax(z)      

        loss = cross_entropy_loss(y_encoded, y_pred)
        losses.append(loss)

        m = X.shape[0]
        dz = (y_pred - y_encoded) / m    
        dW = np.dot(X.T, dz)            
        db = np.sum(dz, axis=0, keepdims=True)

        W -= learning_rate * dW
        b -= learning_rate * db

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")
    
    return W, b, losses

def predict(X, W, b):
    z = np.dot(X, W) + b
    y_pred = softmax(z)
    return np.argmax(y_pred, axis=1)

W, b, losses = train(X, y_encoded, W, b, learning_rate=0.1, epochs=1000)

y_pred = predict(X, W, b)

accuracy = np.mean(y_pred == y)
print(f"Training Accuracy: {accuracy*100:.2f}%")

def precision_recall_f1(y_true, y_pred, num_classes):
    precisions = []
    recalls = []
    f1s = []

    for cls in range(num_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precision_avg = np.mean(precisions)
    recall_avg = np.mean(recalls)
    f1_avg = np.mean(f1s)

    return precision_avg, recall_avg, f1_avg

precision, recall, f1 = precision_recall_f1(y, y_pred, num_classes)
print(f"Custom Model:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X, y)

y_pred_sklearn = model.predict(X)

acc = accuracy_score(y, y_pred_sklearn)
prec = precision_score(y, y_pred_sklearn, average='macro')
rec = recall_score(y, y_pred_sklearn, average='macro')
f1 = f1_score(y, y_pred_sklearn, average='macro')

print("\nSklearn Model:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

