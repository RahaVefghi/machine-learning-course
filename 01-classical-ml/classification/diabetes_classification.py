import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import os

def load_data():
  base_dir = os.path.dirname(os.path.abspath(__file__))
  data_path = os.path.join(base_dir, "data", "diabetes.csv")
  dataset = pd.read_csv(data_path)
  # rows, cols = dataset.shape
  # print(f"number of rows:{rows} number of cols:{cols}")

  zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI']

  for column in zero_not_accepted:
    
    dataset[column] = dataset[column].replace(0, np.nan)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.nan , mean)
    
  X = dataset.iloc[: , 0:8]
  Y = dataset.iloc[: , 8]
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  
  sc_x = StandardScaler()
  x_train = sc_x.fit_transform(x_train) 
  x_test = sc_x.transform(x_test)
  return x_train, x_test, y_train, y_test

def algorithm():  
  #classifier = KNeighborsClassifier()
  #classifier = LogisticRegression() 
  classifier = SGDClassifier(loss="log_loss")
  
  classifier.fit(x_train, y_train)
  #print(classifier.coef_)
  #print(classifier.intercept_)
  return classifier

def show_result():
  y_pred = classifier.predict(x_test)
  acc = accuracy_score(y_pred, y_test)
  print("accuracy: {:.2f}".format(acc*100))
  acc2 = classifier.score(x_test, y_test)
  #print("accuracy is : " , acc2 * 100)
  prob = classifier.predict_proba(x_test)
  print(prob)
  

x_train, x_test, y_train, y_test = load_data()
classifier = algorithm()
show_result()


