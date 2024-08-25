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

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=4, max_features='sqrt'),
}

results = {}

for name, model in models.items():
    r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    mae_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    rmse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')

    results[name] = {
        "R2 (CV Mean)": np.mean(r2_scores),
        "MAE (CV Mean)": -np.mean(mae_scores),
        "RMSE (CV Mean)": -np.mean(rmse_scores)
    }

print(pd.DataFrame(results).T)