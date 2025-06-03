import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def logistic_regression_classification(file_path):
    data = pd.read_csv(file_path)
    data['Rain'] = data['Rain'].map({'rain': 1, 'no rain': 0})
    shuffled_data = shuffle(data, random_state=0)
    x = shuffled_data[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
    y = shuffled_data['Rain']

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    model = LogisticRegression().fit(X_train, y_train)

    # Training predictions and metrics
    train_pred = model.predict(X_train)
    print("Training set Results:")
    print("Accuracy:", accuracy_score(y_train, train_pred))
    print("Precision:", precision_score(y_train, train_pred, average='binary'))
    print("Recall:", recall_score(y_train, train_pred, average='binary'))
    print("F1 Score:", f1_score(y_train, train_pred, average='binary'))

    # Testing predictions and metrics
    test_pred = model.predict(X_test)
    print("\nTesting set Results:")
    print("Accuracy:", accuracy_score(y_test, test_pred))
    print("Precision:", precision_score(y_test, test_pred, average='binary'))
    print("Recall:", recall_score(y_test, test_pred, average='binary'))
    print("F1 Score:", f1_score(y_test, test_pred, average='binary'))

    return model

def main():
    file_path = "datasets/weather_forecast_data.csv"  
    logistic_regression_classification(file_path)

if __name__ == "__main__":
    main()
