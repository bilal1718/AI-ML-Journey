import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/CT/Desktop/AI_Journey/ML_Projects/California_Housing_Price/housing.csv")
# print(data["ocean_proximity"].value_counts())
# print(data.head(20))
# print(data.describe())


# data.hist(bins=50, figsize=(12,8))
# plt.show()


 # Custom shuffling code
# def shuffle_and_split_data(data, test_size):
#     shuffled_indices= np.random.permutation(len(data))
#     test_set_size=int(len(data) * test_size)
#     test_indices=shuffled_indices[:test_set_size]
#     train_indices=shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set , test_set = shuffle_and_split_data(data, 0.2)

from sklearn.model_selection import train_test_split

train_set, test_set=train_test_split(data, test_size=0.2, random_state=42)

data["income_cat"]=pd.cut(
    data["median_income"],
    bins=[0.,1.5,3.0,4.5,6.,np.inf],
    labels=[1,2,3,4,5]
)

data["income_cat"].value_counts().sort_index().plot.bar(rot=0,grid=True)
plt.xlabel("Income Category")
plt.ylabel("Number of districts")
plt.show()