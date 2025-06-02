import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('datasets/Salary_dataset.csv')
data.drop(columns=['Unnamed: 0'], inplace=True)

x=data["YearsExperience"]
y=data["Salary"]

# print(data.info())


def prediction(x,w,b):
    return w*x + b

def compute_cost(x,y,w,b, m):
    y_pred=w*x + b
    take_sum=np.sum((y_pred - y) ** 2)
    return (1/(2*m)) * take_sum

def batch_gradient_descent(x,y,w,b,m,num_iter,alpha):
    cost_history=np.ones(num_iter)
    for i in range(num_iter):
        y_pred=w*x + b
        d_dw=(1/m)*np.sum((y_pred - y) * x)
        d_db=(1/m)*np.sum((y_pred - y))
        w=w-(alpha * d_dw)
        b=b-(alpha * d_db)
        cost_history[i]=compute_cost(x,y,w,b,m)
    return w,b,cost_history


w=0
b=0
m=x.count()
num_iter=1000
alpha=0.01

w,b,cost_history=batch_gradient_descent(x,y,w,b,m,num_iter,alpha)

y_pred=prediction(x,w,b)

print("Final weight (w):", w)
print("Final bias (b):", b)
print("Predicted salaries:\n", y_pred)

plt.plot(range(num_iter), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function using Batch Gradient Descent')
plt.show()


plt.scatter(x, y , color='black')
plt.plot(x, y_pred, color='red')
plt.legend(['Real data', 'Regression Line'])
plt.xlabel('Years Of experience')
plt.ylabel('Salary')
plt.title('Years Of experience vs Salary')
plt.show()