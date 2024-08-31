import pandas as pd

data = pd.read_csv("C:/Users/CT/Desktop/AI_Journey/ML_Projects/California_Housing_Price/housing.csv")
print(data["ocean_proximity"].value_counts())
print(data.head(20))
print(data.describe())