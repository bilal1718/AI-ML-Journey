import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = pd.read_csv('GDA_SVM_NB_SR/GDA/diabetes.csv')
X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness',
          'Insulin','BMI','DiabetesPedigreeFunction','Age']].values
y = data['Outcome'].values

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def compute_parameters(X, y):
    m, n = X.shape
    phi = np.mean(y)

    X0 = X[y == 0]
    X1 = X[y == 1]
    
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    
    sigma = ((X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)) / m
    
    return phi, mu0, mu1, sigma

def gaussian_likelihood(x, mu, sigma_inv, sigma_det):
    d = x - mu
    return -0.5 * (np.log(sigma_det) + d @ sigma_inv @ d)

def predict(X, phi, mu0, mu1, sigma):
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    
    y_pred = []
    for x in X:
        log_p0 = gaussian_likelihood(x, mu0, sigma_inv, sigma_det) + np.log(1 - phi)
        log_p1 = gaussian_likelihood(x, mu1, sigma_inv, sigma_det) + np.log(phi)
        y_pred.append(1 if log_p1 > log_p0 else 0)
    
    return np.array(y_pred)

phi, mu0, mu1, sigma = compute_parameters(X, y)
y_pred_scratch = predict(X, phi, mu0, mu1, sigma)

print("GDA From Scratch:")
print(f"Accuracy : {accuracy_score(y, y_pred_scratch):.4f}")
print(f"Precision: {precision_score(y, y_pred_scratch):.4f}")
print(f"Recall   : {recall_score(y, y_pred_scratch):.4f}")
print(f"F1 Score : {f1_score(y, y_pred_scratch):.4f}\n")

model = LinearDiscriminantAnalysis()
model.fit(X, y)
y_pred_sklearn = model.predict(X)

print("Sklearn LDA:")
print(f"Accuracy : {accuracy_score(y, y_pred_sklearn):.4f}")
print(f"Precision: {precision_score(y, y_pred_sklearn):.4f}")
print(f"Recall   : {recall_score(y, y_pred_sklearn):.4f}")
print(f"F1 Score : {f1_score(y, y_pred_sklearn):.4f}")


