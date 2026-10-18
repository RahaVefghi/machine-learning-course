import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score

def load_and_preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "diabetes.csv")
    dataset = pd.read_csv(data_path)

    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_not_accepted:
        dataset[column] = dataset[column].replace(0, np.nan)
        mean_val = int(dataset[column].mean(skipna=True))
        dataset[column] = dataset[column].replace(np.nan, mean_val)

    X = dataset.iloc[:, 0:8]
    y = dataset.iloc[:, 8]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test

def train_classifier(x_train, y_train, loss="log_loss"):
    classifier = SGDClassifier(loss=loss, random_state=42)
    classifier.fit(x_train, y_train)
    return classifier

def evaluate_classifier(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(x_test[:5])
        print("Sample predicted probabilities (first 5 samples):\n", probs)

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_and_preprocess_data()
    model = train_classifier(x_train, y_train)
    evaluate_classifier(model, x_test, y_test)
