import numpy as np
import pandas as pd
import os

os.chdir("c:/Users/CT/Desktop/AI_Journey/ML_Projects/Customer_Insights_And_Prediction/")
df = pd.read_csv("sales_forecasting_dataset.csv")
df.drop(columns=["Row ID","Order ID","Product Name","Customer ID","Customer Name","Country","Product ID","Postal Code"],inplace=True)


X=df.drop("Sales",axis=1)
y=df["Sales"]
cols_to_encode = X
for col in cols_to_encode:
    freq_encoding = X[col].value_counts() / len(X)
    X[col] = X[col].map(freq_encoding)

# print("Input Features: ")
# print(X.head())
# print("Target Value: ")
# print(y.head())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lr=LinearRegression()
lr.fit(X,y)
print(r2_score(lr.predict(X), y))




# The data is from country => United States