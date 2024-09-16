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
data=strat_train_set.copy()
# data.plot(kind="scatter" , x="longitude", y="latitude", grid=True,
#            s=data["population"] / 100 , label="population", c="median_house_value",
#              cmap="jet" , colorbar=True , legend=True, sharex=False, figsize=(10,7))
# plt.show()


# corr_matrix=data.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))


data["rooms_per_house"]=data["total_rooms"] / data["households"]
data["bedrooms_ratio"]=data["total_bedrooms"] / data["total_rooms"]
data["people_per_house"]=data["population"] / data["households"]

# corr_matrix=data.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

data=strat_train_set.drop("median_house_value", axis=1)
data_labels=strat_train_set["median_house_value"].copy()


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

imputer=SimpleImputer(strategy="median")
data_num=data.select_dtypes(include=[np.number])
imputer.fit(data_num)

X=imputer.transform(data_num)
data_tr=pd.DataFrame(X, columns=data_num.columns, index=data_num.index)
data_cat=data[["ocean_proximity"]]
ordinal_encoder=OrdinalEncoder()
data_cat_encoded=ordinal_encoder.fit_transform(data_cat)
print(data_cat_encoded[:8])

cat_encoder=OneHotEncoder()
data_cat_1hot=cat_encoder.fit_transform(data_cat)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

std_scalar=StandardScaler()
data_num_std_scaled=std_scalar.fit_transform(data_num)

from sklearn.metrics.pairwise import rbf_kernel
age_simil_35=rbf_kernel(data[["housing_median_age"]], [[35]], gamma=0.1)

target_scalar=StandardScaler()
scaled_labels=target_scalar.fit_transform(data_labels.to_frame())

model=LinearRegression()
model.fit(data[["median_income"]], scaled_labels)
some_new_data=data[["median_income"]].iloc[:5]

scaled_predictions=model.predict(some_new_data)
predictions=target_scalar.inverse_transform(scaled_predictions)

model=TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(data[["median_income"]], data_labels)
predictions=model.predict(some_new_data)
print(predictions)

from sklearn.preprocessing import FunctionTransformer

log_transformer=FunctionTransformer(np.log, inverse_func=np.exp)
log_pop=log_transformer.transform(data[["population"]])


rbf_transformer=FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35=rbf_transformer.transform(data[["housing_median_age"]])

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters=n_clusters
        self.gamma=gamma
        self.random_state=random_state
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans=KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X, sample_weight=sample_weight)
        return self
    def transform(self, X):
        return rbf_kernel(X, self.kmeans.cluster_centers_,gamma=self.gamma)
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
cluster_simil=ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities=cluster_simil.fit_transform(data[["latitude", "longitude"]], sample_weight=data_labels)

print(similarities[:3].round(2))

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

num_pipeline=Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("standardize",StandardScaler()),
])
num_pipeline=make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
data_num_prepared=num_pipeline.fit_transform(data_num)
print(data_num_prepared[:2].round(2))
df_data_num_prepared=pd.DataFrame(data_num_prepared, columns=num_pipeline.get_feature_names_out(), index=data_num.index)

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
num_attribs=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
cat_attribs=["ocean_proximity"]

cat_pipeline=make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing=ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

preprocessing=make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

data_prepared=preprocessing.fit_transform(data)

def column_ratio(X):
    return X[:,[0]]/X[:,[1]]
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )
log_pipeline=make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)
cluster_simil=ClusterSimilarity(n_clusters=10, gamma=1. , random_state=42)
default_num_pipeline=make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

preprocessing=ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("gep", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
remainder=default_num_pipeline
)

data_prepared=preprocessing.fit_transform(data)
print(data_prepared.shape)

print(preprocessing.get_feature_names_out())

from sklearn.linear_model import LinearRegression

lin_reg=make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(data, data_labels)

data_predictions=lin_reg.predict(data)
print(data_predictions[:5].round(-2))
print(data_labels.iloc[:5].values)

from sklearn.metrics import mean_squared_error

lin_rmse=mean_squared_error(data_labels, data_predictions, squared=False)
print(lin_rmse)

from sklearn.tree import DecisionTreeRegressor

tree_reg=make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(data, data_labels)

data_predictions=tree_reg.predict(data)
tree_rmse=mean_squared_error(data_labels, data_predictions, squared=False)
print(tree_rmse)

from sklearn.model_selection import cross_val_score

tree_rmses=-cross_val_score(tree_reg, data, data_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rmses).describe())

from sklearn.ensemble import RandomForestRegressor

forest_reg=make_pipeline(preprocessing, RandomForestRegressor(random_state=42))

forest_rmses=-cross_val_score(forest_reg, data, data_labels, scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(forest_rmses).describe())

from sklearn.model_selection import GridSearchCV

full_pipeline=Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid=[
    {'preprocessing__geo__n_clusters':[5,8,10],
     'random_forest__max_features':[4,6,8]},
    {'preprocessing__geo__n_clusters':[10,15],
     'random_forest__max_features':[6,8,10]},
]
grid_search=GridSearchCV(full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
grid_search.fit(data, data_labels)
print(grid_search.best_params_)
cv_res=pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
print(cv_res.head())


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs=[
    {'preprocessing__geo__n_clusters':randint(low=3, high=50),
     'random_forest__max_features':randint(low=2, high=20)},
]

