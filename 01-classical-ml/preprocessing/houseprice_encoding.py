import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

def load_house_attributes(file_path):
    cols = ["bedrooms", "bathroom", "area", "zipcode", "price"]
    df = pd.read_csv(file_path, sep=" ", header=None, names=cols)

    # Filter out infrequent zipcodes with less than 25 occurrences
    zipcode_counts = df["zipcode"].value_counts()
    frequent_zipcodes = zipcode_counts[zipcode_counts >= 25].index
    df_filtered = df[df["zipcode"].isin(frequent_zipcodes)].copy()

    return df_filtered

def preprocess_features(train, test):
    continuous_features = ["bedrooms", "bathroom", "area"]

    # Scale continuous features
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(train[continuous_features])
    test_cont = scaler.transform(test[continuous_features])

    # One-Hot / Binary encode categorical zipcode feature
    binarizer = LabelBinarizer()
    train_cat = binarizer.fit_transform(train["zipcode"])
    test_cat = binarizer.transform(test["zipcode"])

    # Combine continuous and categorical features
    x_train = np.hstack([train_cont, train_cat])
    x_test = np.hstack([test_cont, test_cat])

    return x_train, x_test

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "HousesInfo.txt")

    df = load_house_attributes(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    x_train, x_test = preprocess_features(train_df, test_df)

    # Normalize target prices by maximum training price
    max_price = train_df["price"].max()
    y_train = train_df["price"] / max_price
    y_test = test_df["price"] / max_price

    model = LinearRegression()
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, preds)

    print(f"Houses processed: {len(df)}")
    print(f"Features: {x_train.shape[1]} (3 continuous + {x_train.shape[1] - 3} encoded zipcodes)")
    print(f"Model Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

if __name__ == "__main__":
    main()
