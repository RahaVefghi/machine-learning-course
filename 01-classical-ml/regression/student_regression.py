import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "student_scores.csv")
    dataset = pd.read_csv(data_path)

    X = dataset.iloc[:, :1]
    y = dataset.iloc[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test

def train_and_evaluate():
    x_train, x_test, y_train, y_test = load_data()

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    print(f"Fitted Model: y = {regressor.coef_[0]:.2f}x + {regressor.intercept_:.2f}")

    y_pred = regressor.predict(x_test)
    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": np.round(y_pred, 2)})
    print("\nSample Predictions:")
    print(results_df.head(5))

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\nEvaluation Metrics:")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_and_evaluate()
