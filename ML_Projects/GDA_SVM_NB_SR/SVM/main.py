import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

data = pd.read_csv("SVM/heart_failure_clinical_records_dataset.csv")

X = data[['ejection_fraction', 'serum_creatinine','age','anaemia','creatinine_phosphokinase','diabetes',
          'high_blood_pressure','platelets','serum_sodium','sex','smoking','time']].values
y = data['DEATH_EVENT'].values


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

y = np.where(y == 0, -1, 1)

print("Features after normalization:\n", X[:5])
print("Labels:\n", y[:20])


w = np.zeros(X.shape[1])  
b = 0
lr = 0.001 
epochs = 1000
C = 10  

n = len(y)

for epoch in range(epochs):
    for i in range(n):
        xi = X[i]
        yi = y[i]
        
        condition = yi * (np.dot(w, xi) + b) >= 1
        
        if condition:
            dw = w
            db = 0
        else:
            dw = w - C * yi * xi
            db = -C * yi
        
        w = w - lr * dw
        b = b - lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | w: {w}, b: {b}")

def predict(X):
    return np.sign(np.dot(X, w) + b)

y_pred = predict(X)


f1 = f1_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred)
accuracy = np.mean(y_pred == y)

print("Training accuracy:", accuracy)
print("Training f1:", f1)
print("Training precision:", precision)
print("Training recall:", recall)


