from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

original_dataset = pd.read_csv('C:/Users/hp/Desktop/2015-building-energy-benchmarking.csv')

columns_to_keep = [
    'BuildingType', 'PrimaryPropertyType', 'YearBuilt', 'NumberofBuildings',
    'NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking',
    'PropertyGFABuilding', 'SiteEnergyUse', 'Electricity',
    'NaturalGas', 'SiteEUI(kBtu/sf)'
]

cleaned_dataset = original_dataset[columns_to_keep].dropna()

categorical_cols = cleaned_dataset.select_dtypes(include=['object']).columns

category_to_int_map = {}
for col in categorical_cols:
    unique_categories = cleaned_dataset[col].unique()
    category_to_int_map[col] = {category: idx for idx, category in enumerate(unique_categories)}
    cleaned_dataset[col] = cleaned_dataset[col].map(category_to_int_map[col])

X = cleaned_dataset.drop(columns=["SiteEUI(kBtu/sf)"])
y = cleaned_dataset["SiteEUI(kBtu/sf)"]

def scaled_features(X_input, X_mean, X_std):
    return (X_input - X_mean) / X_std

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = scaled_features(X, X_mean, X_std)

def train_linear_regression(X, y, alpha=0.001, num_iters=2000):
    b = 0
    m = X.shape[0]
    w = np.zeros(X.shape[1])
    cost_history = []

    for _ in range(num_iters):
        y_pred = np.dot(X, w) + b
        cost = y_pred - y
        cost_value = np.mean(cost ** 2) / 2
        cost_history.append(cost_value)
        df_dw = (1/m) * np.dot(X.T, cost)
        df_db = (1/m) * np.sum(cost)
        w = w - (alpha * df_dw)
        b = b - (alpha * df_db)
    return w, b, cost_history

w, b, cost_history = train_linear_regression(X_scaled.to_numpy(), y.to_numpy())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input_df = pd.DataFrame([data])
    print("User Input:")
    print(user_input_df)

    for col in categorical_cols:
        if col in user_input_df.columns:
            user_input_df[col] = user_input_df[col].map(lambda x: category_to_int_map[col].get(x, -1))

    for col in X.columns:
        if col not in user_input_df.columns:
            user_input_df[col] = 0

    user_input_df = user_input_df[X.columns]
    user_input_scaled = scaled_features(user_input_df, X_mean, X_std)
    print("Processes User Input:")
    print(user_input_df)

    prediction = np.dot(user_input_scaled.to_numpy(), w) + b
    return jsonify({'prediction': prediction[0]})
@app.route('/get-historical-data', methods=['GET'])
def get_historical_data():
    historical_data = cleaned_dataset[['SiteEUI(kBtu/sf)',
                                        'PropertyGFATotal', 
                                        'YearBuilt', 'NumberofBuildings', 'NumberofFloors',
                                          'Electricity', 'NaturalGas',
                                            'PrimaryPropertyType','BuildingType',
                                           'PropertyGFAParking','PropertyGFABuilding',
                                             'SiteEnergyUse' ]].to_dict(orient='list')
    return jsonify(historical_data)

if __name__ == '__main__':
    app.run(debug=True)
