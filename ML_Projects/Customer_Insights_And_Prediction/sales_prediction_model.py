import numpy as np
import pandas as pd
import os

os.chdir("c:/Users/CT/Desktop/AI_Journey/ML_Projects/Customer_Insights_And_Prediction/")
df = pd.read_csv("sales_forecasting_dataset.csv")
df.drop(columns=["Row ID","Order ID","Product Name","Customer ID","Customer Name","Country","Product ID"],inplace=True)

print(df.head())






# The data is from country => United States