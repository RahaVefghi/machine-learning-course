import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, zero_one_loss

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "iris.data")
    dataset = pd.read_csv(
        data_path,
        header=None,
        names=['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    )
    X = dataset.iloc[:, :4]
    y = dataset.iloc[:, 4]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(x_train, y_train, n_neighbors=5):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train, y_train)
    return clf

def evaluate_model(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    loss = zero_one_loss(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Zero-One Loss: {loss * 100:.2f}%")

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    clf = train_model(x_train, y_train)
    evaluate_model(clf, x_test, y_test)
