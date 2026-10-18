import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "petrol_consumption.csv")
    dataset = pd.read_csv(data_path)

    features = ['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licens']
    target = 'Petrol_Consumption'

    X = dataset[features]
    y = dataset[target]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, features

def train_and_compare():
    x_train, x_test, y_train, y_test, feature_names = load_data()

    # 1. Ordinary Least Squares (Linear Regression)
    ols = LinearRegression()
    ols.fit(x_train, y_train)
    y_pred_ols = ols.predict(x_test)

    # 2. Stochastic Gradient Descent Regressor
    sgd = SGDRegressor(max_iter=1000, random_state=42, eta0=0.001)
    sgd.fit(x_train, y_train)
    y_pred_sgd = sgd.predict(x_test)

    # Coefficients comparison
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "OLS_Coefficient": np.round(ols.coef_, 4),
        "SGD_Coefficient": np.round(sgd.coef_, 4)
    })
    print("Model Coefficients Comparison:")
    print(coef_df)

    print("\nModel Evaluation (Linear Regression):")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred_ols):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ols)):.2f}")

    print("\nModel Evaluation (SGD Regressor):")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred_sgd):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_sgd)):.2f}")

if __name__ == "__main__":
    train_and_compare()