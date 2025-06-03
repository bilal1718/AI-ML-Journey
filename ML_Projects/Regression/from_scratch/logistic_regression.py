import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def shuffle_data(data):
    return data.sample(frac=1).reset_index(drop=True)

def train_test_split(data, test_size):
    split_point = int(len(data) * (1 - test_size))
    train_data=data[:split_point]
    test_data=data[split_point:]
    return train_data, test_data

data=pd.read_csv('datasets/weather_forecast_data.csv')
data['Rain']=data['Rain'].map({'rain':1, 'no rain':0})


shuffled_data=shuffle_data(data)
train_data , test_data=train_test_split(shuffled_data, 0.2)

train_x=train_data[['Temperature','Humidity','Wind_Speed','Cloud_Cover','Pressure']]
train_y=train_data['Rain']

train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)

train_x=train_x.values
train_y=train_y.values

print('Rain Days: ', train_y.sum()) 
print('No Rain Days',len(train_y) - train_y.sum())


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1/(1+np.exp(-z))

def cost_function(x, y, w, b, m):
    z = np.dot(x, w) + b
    h = sigmoid(z)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    return (1/m) * np.sum(-y*np.log(h) - (1 - y)*np.log(1 - h))

def gradient_descent(x, y, w, b, m, num_iter, alpha):
    cost_history = np.zeros(num_iter)
    for i in range(num_iter):
        z = np.dot(x, w) + b
        h = sigmoid(z)
        d_dw = (1/m) * (x.T @ (h - y))
        d_db = (1/m) * np.sum(h - y)
        w = w - alpha * d_dw
        b = b - alpha * d_db
        cost_history[i] = cost_function(x, y, w, b, m)
    return w, b, cost_history

num_iter=1000
alpha=0.01
w=np.zeros(train_x.shape[1])
b=0
m = train_x.shape[0]

w,b, cost_history=gradient_descent(train_x, train_y, w, b, m, num_iter, alpha)


# Taesting on training data

prediction=sigmoid(np.dot(train_x, w) + b)

class_preds = (prediction >= 0.3).astype(int)
accuracy = np.mean(class_preds == train_y)
print("Accuracy:", accuracy)

TP = np.sum((class_preds == 1) & (train_y == 1))
TN = np.sum((class_preds == 0) & (train_y == 0))
FP = np.sum((class_preds == 1) & (train_y == 0))
FN = np.sum((class_preds == 0) & (train_y == 1))

epsilon = 1e-15

precision = TP / (TP + FP + epsilon)
recall = TP / (TP + FN + epsilon)
f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)




# print(prediction)

plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.grid(True)
plt.show()


test_x=test_data[['Temperature','Humidity','Wind_Speed','Cloud_Cover','Pressure']]
test_y=test_data['Rain']

test_x = (test_x - train_data[['Temperature','Humidity','Wind_Speed','Cloud_Cover','Pressure']].mean()) / \
         train_data[['Temperature','Humidity','Wind_Speed','Cloud_Cover','Pressure']].std()


print('Rain Days in test set: ', test_y.sum()) 
print('No Rain Days in test set',len(test_y) - test_y.sum())
# Testing on test data

prediction=sigmoid(np.dot(test_x, w) + b)

class_preds = (prediction >= 0.35).astype(int)
accuracy = np.mean(class_preds == test_y)
print("Accuracy in test set:", accuracy)

TP = np.sum((class_preds == 1) & (test_y == 1))
TN = np.sum((class_preds == 0) & (test_y == 0))
FP = np.sum((class_preds == 1) & (test_y == 0))
FN = np.sum((class_preds == 0) & (test_y == 1))

epsilon = 1e-15

precision = TP / (TP + FP + epsilon)
recall = TP / (TP + FN + epsilon)
f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

print("Precision in test set:", precision)
print("Recall in test set:", recall)
print("F1 Score in test set:", f1_score)