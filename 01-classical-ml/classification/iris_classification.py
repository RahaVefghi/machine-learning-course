import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss

import os

def load_data():
  base_dir = os.path.dirname(os.path.abspath(__file__))
  data_path = os.path.join(base_dir, "data", "iris.data")
  dataset = pd.read_csv(data_path,
                      header=None,
                      names=['sepal length', 'sepal width', 'petal length', 'petal width', 'label'])
  rows, cols = dataset.shape
# print(f"numbers of rows :{rows} , number of cols: {cols}")
#print("number of samples : {}, number of features: {}". format(rows, cols))

  data = dataset.iloc[:, :4]
  label = dataset.iloc[: , 4]
  x_trian, x_test, y_train, y_test = train_test_split(data, label, test_size= 0.2, random_state=42)
  
  return x_trian, x_test, y_train, y_test

def training():
  clf= KNeighborsClassifier(n_neighbors=5)
  clf.fit(x_trian, y_train)
  return clf

def result():
  y_pred = clf.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  print("accuracy: {:.2f}".format(acc*100 ))
  loss = zero_one_loss(y_test, y_pred)
  # print(f"loss:{loss*100}")
  print("loss:{:.2f}".format(loss*100))
   

x_trian, x_test, y_train, y_test  = load_data()
clf = training() 
result()

# print(y_test)
# print(y_pred)


