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
data["income_cat"]=pd.cut(
    data["median_income"],
    bins=[0.,1.5,3.0,4.5,6.,np.inf],
    labels=[1,2,3,4,5]
)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

train_set, test_set=train_test_split(data, stratify=data["income_cat"], test_size=0.2, random_state=42)
splitter=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits=[]
for train_index, test_index in splitter.split(data, data["income_cat"]):
    strat_train_set_n=data.iloc[train_index]
    strat_test_set_n=data.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
strat_train_set, strat_test_set=strat_splits[0]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# data["income_cat"].value_counts().sort_index().plot.bar(rot=0,grid=True)
# plt.xlabel("Income Category")
# plt.ylabel("Number of districts")
# plt.show()