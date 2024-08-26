import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir("c:/Users/CT/Desktop/AI_Journey/ML_Projects/Customer_Insights_And_Prediction/")
df = pd.read_csv("sales_forecasting_dataset.csv")
df.drop(columns=["Row ID","Order ID","Product Name","Customer ID","Customer Name","Country","Product ID","Postal Code"],inplace=True)
cols_to_encode = df
for col in cols_to_encode:
    freq_encoding = df[col].value_counts() / len(df)
    df[col] = df[col].map(freq_encoding)
X=df.drop("Sales",axis=1)
y=df["Sales"]
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=4, max_features='sqrt')
model.fit(X_train, y_train)
def get_user_input():
    user_input = {}
    user_input['Order Date'] = input("Enter Order Date (YYYY-MM-DD): ")
    user_input['Ship Date'] = input("Enter Ship Date (YYYY-MM-DD): ")
    user_input['Ship Mode'] = input("Enter Ship Mode: ")
    user_input['Segment'] = input("Enter Segment: ")
    user_input['City'] = input("Enter City: ")
    user_input['State'] = input("Enter State: ")
    user_input['Region'] = input("Enter Region: ")
    user_input['Category'] = input("Enter Category: ")
    user_input['Sub-Category'] = input("Enter Sub-Category: ")
    
    return pd.DataFrame([user_input])
def transform_input(user_df, reference_df):
    for col in user_df.columns:
        freq_encoding = reference_df[col].value_counts() / len(reference_df)
        user_df[col] = user_df[col].map(freq_encoding)
    
    return user_df
def predict_sales(user_df):
    user_df_transformed = transform_input(user_df, df)
    
    prediction = model.predict(user_df_transformed)
    
    return prediction[0]


user_df = get_user_input()
predicted_sales = predict_sales(user_df)
print(f"Predicted Sales: {predicted_sales}")

