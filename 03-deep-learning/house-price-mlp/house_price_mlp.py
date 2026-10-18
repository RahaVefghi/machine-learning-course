import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import mean_absolute_percentage_error
from keras import models, layers

EPOCHS = 100
BATCH_SIZE = 32

def load_house_attributes(file_path):
    cols = ["bedrooms", "bathroom", "area", "zipcode", "price"]
    df = pd.read_csv(file_path, sep=" ", header=None, names=cols)

    # Filter out zipcodes with fewer than 25 samples
    zipcode_counts = df["zipcode"].value_counts()
    frequent_zipcodes = zipcode_counts[zipcode_counts >= 25].index
    return df[df["zipcode"].isin(frequent_zipcodes)].copy()

def preprocess_features(train, test):
    continuous_features = ["bedrooms", "bathroom", "area"]

    scaler = StandardScaler()
    train_cont = scaler.fit_transform(train[continuous_features])
    test_cont = scaler.transform(test[continuous_features])

    binarizer = LabelBinarizer()
    train_cat = binarizer.fit_transform(train["zipcode"])
    test_cat = binarizer.transform(test["zipcode"])

    x_train = np.hstack([train_cont, train_cat])
    x_test = np.hstack([test_cont, test_cat])
    return x_train, x_test

def build_and_train_mlp(x_train, y_train, x_test, y_test):
    model = models.Sequential([
        layers.Input(shape=(x_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model.summary()

    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )
    return model, history

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "HousesInfo.txt")

    df = load_house_attributes(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    x_train, x_test = preprocess_features(train_df, test_df)

    max_price = train_df["price"].max()
    y_train = (train_df["price"] / max_price).values
    y_test = (test_df["price"] / max_price).values

    model, history = build_and_train_mlp(x_train, y_train, x_test, y_test)

    preds = model.predict(x_test).flatten()
    mape = mean_absolute_percentage_error(y_test, preds)

    print(f"\n[EVALUATION] Test MSE Loss: {model.evaluate(x_test, y_test, verbose=0):.4f}")
    print(f"[EVALUATION] Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

if __name__ == "__main__":
    main()
