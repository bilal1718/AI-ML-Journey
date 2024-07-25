import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

original_dataset = pd.read_csv('C:/Users/hp/Desktop/2015-building-energy-benchmarking.csv')
columns_to_keep = [
    'BuildingType', 'PrimaryPropertyType', 'YearBuilt', 'NumberofBuildings', 
    'NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking', 
    'PropertyGFABuilding(s)', 'SiteEnergyUse(kBtu)', 'Electricity(kWh)', 
    'NaturalGas(therms)', 'SiteEUI(kBtu/sf)'
]
cleaned_dataset = original_dataset[columns_to_keep].copy()
cleaned_dataset = cleaned_dataset.dropna()
cleaned_file_path = 'C:/Users/hp/Desktop/cleaned_2015_building_energy_benchmarking.csv'
cleaned_dataset.to_csv(cleaned_file_path, index=False)

dataset = pd.read_csv(cleaned_file_path)

categorical_cols = dataset.select_dtypes(include=['object']).columns

def label_encode(dataset, column):
    unique_categories = dataset[column].unique()
    category_to_int = {category: idx for idx, category in enumerate(unique_categories)}
    dataset[column + '_Encoded'] = dataset[column].map(category_to_int)
    return dataset

for col in categorical_cols:
    dataset = label_encode(dataset, col)

encoded_columns = [col + '_Encoded' for col in categorical_cols]
dataset = dataset.drop(columns=categorical_cols)

encoded_file_path = 'C:/Users/hp/Desktop/encoded_dataset.csv'
dataset.to_csv(encoded_file_path, index=False)

X = dataset.drop(columns=["SiteEUI(kBtu/sf)"])
y = dataset["SiteEUI(kBtu/sf)"]
def scaled_features(X_input):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scal=(X_input - X_mean) / X_std
    return X_scal

X_scaling = scaled_features(X)
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

w, b, cost_history = train_linear_regression(X_scaling.to_numpy(), y.to_numpy())

# Plot cost history
# plt.figure(figsize=(8, 5))
# plt.plot(range(len(cost_history)), cost_history)
# plt.xlabel('Iterations', fontsize=12)
# plt.ylabel('Cost', fontsize=12)
# plt.title('Cost Function History', fontsize=14)
# plt.show()

def predict_from_user_input(user_input):
    user_input_df = pd.DataFrame([user_input], columns=X.columns)
    for col in categorical_cols:
        if col in user_input_df.columns:
            user_input_df = label_encode(user_input_df, col)
    for col in X.columns:
        if col not in user_input_df.columns:
            user_input_df[col] = 0
    user_input_df = user_input_df[X.columns]
    user_input_scaled = scaled_features(user_input_df)
    prediction = np.dot(user_input_scaled.to_numpy(), w) + b
    return prediction[0]


user_input = {
    'BuildingType_Encoded': 1,
    'PrimaryPropertyType_Encoded': 2,
    'YearBuilt': 1995,
    'NumberofBuildings': 1,
    'NumberofFloors': 8,
    'PropertyGFATotal': 10000,
    'PropertyGFAParking': 10000,
    'PropertyGFABuilding(s)': 40000,
    'SiteEnergyUse(kBtu)': 10000,
    'Electricity(kWh)': 100000,
    'NaturalGas(therms)': 50000
}
y_pred = np.dot(X_scaling.to_numpy(), w) + b
def mean_absolute_error(y_true, y_pred):
    errors = [abs(pred - true) for true, pred in zip(y_true, y_pred)]
    return sum(errors) / len(errors)


prediction = predict_from_user_input(user_input)
print(f"Predicted Site EUI: {prediction:.2f} kBtu/sf")
mae = mean_absolute_error(y, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
correlation_matrix = dataset.corr()
correlation_with_target = correlation_matrix['SiteEUI(kBtu/sf)']
print(correlation_with_target)



# X_scaling = X_scaling.to_numpy()
# features = [
#     'BuildingType_Encoded', 'PrimaryPropertyType_Encoded', 'YearBuilt', 'NumberofBuildings', 
#     'NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking', 
#     'PropertyGFABuilding(s)', 'SiteEnergyUse(kBtu)', 'Electricity(kWh)', 
#     'NaturalGas(therms)'
# ]
# first_half_features = features[:4]
# second_half_features = features[4:8]
# third_half_features = features[8:]
# def plot_scatter_with_trend(features_subset, offset, fig_size, title, num_plots):
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=fig_size)
#     axes = axes.flatten()
#     for i, feature in enumerate(features_subset):
#         ax = axes[i]
#         sample_size = 100
#         sample_indices = np.random.choice(X_scaling.shape[0], sample_size, replace=False)
#         X_sample = X_scaling[sample_indices]
#         y_sample = y.iloc[sample_indices]
#         sns.scatterplot(x=X_sample[:, i + offset], y=y_sample, ax=ax, color='blue', alpha=0.5, label='Actual')
#         sns.lineplot(x=X_sample[:, i + offset], y=np.dot(X_sample, w) + b, ax=ax, color='red', label='Fitted')
#         correlation = np.corrcoef(X_sample[:, i + offset], y_sample)[0, 1]
#         ax.set_title(f'{feature} vs Site Energy Consumption\nCorrelation: {correlation:.2f}', fontsize=10, pad=20)
#         ax.legend()
#         ax.set_xlabel('')
#         ax.set_ylabel('')
#     for j in range(num_plots, len(axes)):
#         fig.delaxes(axes[j])
#     plt.tight_layout(pad=3.0)
#     plt.suptitle(title, fontsize=16, y=1.05)
#     plt.show()

# plot_scatter_with_trend(first_half_features, 0, (20, 10), 'Scatter Plots with Fitted Line (First Set)', 4)
# plot_scatter_with_trend(second_half_features, 4, (20, 10), 'Scatter Plots with Fitted Line (Second Set)', 4)
# plot_scatter_with_trend(third_half_features, 8, (20, 10), 'Scatter Plots with Fitted Line (Third Set)', 3)

# print("Model weights:", w)
# print("Model intercept:", b)
# print(X.head())
# print(y.head())
