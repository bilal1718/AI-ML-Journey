import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
       return data.sample(frac=1).reset_index(drop=True)


def train_test_split(data, test_size=0.2):
       if not 0 < test_size < 1:
           raise ValueError("test_size must be between 0 and 1 (exclusive)")

       split_point = int(len(data) * (1 - test_size))
       train_set = data[:split_point]
       test_set = data[split_point:]
       return train_set, test_set

def prediction(x,w,b):
      return np.dot(x,w) + b

def cost_value(y, w, x, b, m):
    pred_y = np.dot(x, w) + b
    squared_err = np.sum((pred_y - y) ** 2)
    return 1 / (2 * m) * squared_err


def stochistic_gradient_descent(w,b,x,y,m,alpha,num_iter):
      cost_history=np.zeros(num_iter)
      for i in range(num_iter):
        for j in range(m):
            x_i = np.array(x[j], dtype=np.float64)
            y_i= y[j]
            pred_y = np.dot(w,x_i) + b
            d_dw = (1/m) * (pred_y - y_i) * x_i 
            d_db= (1/m) * (pred_y - y_i)
            w=w-(alpha*d_dw)
            b=b-(alpha*d_db)
        cost_history[i]=cost_value(y,w,x,b,m)
      return w,b,cost_history



data = pd.read_csv('datasets/Student_Performance.csv')
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})


shuffled_data = shuffle_data(data)
train_data, test_data = train_test_split(shuffled_data, test_size=0.2)

train_features = train_data[['Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
train_y=train_data['Performance Index']

mean = train_features.mean()
std = train_features.std()
train_features = (train_features - mean) / std


w_init = np.zeros(train_features.shape[1])
b_init = 0
m=train_y.count() 
num_iter=1000
w, b, cost_history=stochistic_gradient_descent(w_init,b_init,train_features.values,train_y.values,m,0.01,num_iter)

predict = prediction(train_features.values, w, b)

print("Final weight (w):", w)
print("Final bias (b):", b)
print("Predicted salaries:\n", predict)

plt.plot(range(num_iter), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function using Stochistic Gradient Descent')
plt.show()


MSE=1/m * (np.sum((predict - train_y)**2))
print("Mean Squared Error for train data : ", MSE)

y_mean=np.mean(train_y)
SS_tot=np.sum((train_y-y_mean)**2)
SS_res=np.sum((train_y-predict)**2)

R2= 1-(SS_res/SS_tot)

print("R² Score for train data:", R2)




test_features = test_data[['Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
test_y = test_data['Performance Index']

test_features = (test_features - mean) / std

test_predictions = prediction(test_features.values, w, b)

m_test = test_y.count()
MSE_test = 1/m_test * np.sum((test_predictions - test_y)**2)
print("Mean Squared Error for test data : ", MSE_test)

y_test_mean = np.mean(test_y)
SS_tot_test = np.sum((test_y - y_test_mean)**2)
SS_res_test = np.sum((test_y - test_predictions)**2)
R2_test = 1 - (SS_res_test / SS_tot_test)
print("R² Score for test data:", R2_test)
