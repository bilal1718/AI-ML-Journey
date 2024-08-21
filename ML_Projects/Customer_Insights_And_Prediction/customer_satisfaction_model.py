import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('future.no_silent_downcasting', True)
os.chdir("c:/Users/CT/Desktop/AI_Journey/ML_Projects/Customer_Insights_And_Prediction/")
df = pd.read_csv("customer_satisfaction_dataset.csv")


X = df[["prices", "reviews.doRecommend", "reviews.numHelpful"]]
y = df["reviews.rating"]
X.loc[:, "reviews.numHelpful"] = X["reviews.numHelpful"].fillna(X["reviews.numHelpful"].median())
X.loc[:, "reviews.doRecommend"] = X["reviews.doRecommend"].fillna(False)
X.loc[:, "reviews.doRecommend"] = X["reviews.doRecommend"].astype(int)
y = y.fillna(y.median()).infer_objects(copy=False)


def extract_price_info(price_list):
    if isinstance(price_list, str): 
        price_list = eval(price_list)
    
    if price_list and isinstance(price_list, list) and len(price_list) > 0:
        first_item = price_list[0] 
        if first_item['currency'] == 'USD':
            return {
                'minPrice': int(first_item['amountMin']),
                'maxPrice': int(first_item['amountMax']),
                'AvgPrice': int((first_item['amountMin'] + first_item['amountMax']) / 2)
            }
    return {'AvgPrice': np.nan}


price_info = X['prices'].apply(extract_price_info)
X = X.join(pd.DataFrame(price_info.tolist()))

X = X.drop(columns=['prices'])

X_filtered = X[X['AvgPrice'] <= 250]
y_filtered = y[X_filtered.index]
X_filtered['Log_AvgPrice'] = np.log1p(X_filtered['AvgPrice'])
X_filtered=X_filtered.drop(columns=['AvgPrice','minPrice','maxPrice'])

X = X_filtered 
y = y_filtered

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


best_params = {
    'bootstrap': False,
    'max_depth': 43,
    'max_features': 'log2',
    'min_samples_leaf': 7,
    'min_samples_split': 8,
    'n_estimators': 185
}
rf_best = RandomForestRegressor(**best_params, random_state=42)
rf_best.fit(X_train, y_train)

y_pred = rf_best.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"Mean Absolute Error: {mae}")

rf_cv_scores = cross_val_score(rf_best, X, y, cv=5, scoring='neg_mean_squared_error')
rf_cv_scores = -rf_cv_scores

# print(f"RandomForestRegressor Cross-Validation MSE Scores: {rf_cv_scores}")
# print(f"Mean MSE: {rf_cv_scores.mean()}")
# print(f"Standard Deviation of MSE: {rf_cv_scores.std()}")
importances = rf_best.feature_importances_
# print(f"Feature Importances: {importances}")


def get_user_input():
    avg_price = float(input("Enter the average price of the product (in USD): "))
    num_helpful = int(input("Enter the number of helpful reviews: "))
    do_recommend = input("Do you recommend the product? (yes/no): ")

    do_recommend = 1 if do_recommend.lower() == 'yes' else 0

    user_data = pd.DataFrame({
        'reviews.doRecommend': [do_recommend],
        'reviews.numHelpful': [num_helpful],
        'Log_AvgPrice': [np.log1p(avg_price)]
    })

    return user_data

def predict_rating(model, user_data):
    predicted_rating = model.predict(user_data)
    return predicted_rating

user_data = get_user_input()
predicted_rating = predict_rating(rf_best, user_data)
print(f"Predicted Rating: {predicted_rating[0]:.2f}")
