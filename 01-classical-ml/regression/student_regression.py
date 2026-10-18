import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler


import os

def load_data():
  base_dir = os.path.dirname(os.path.abspath(__file__))
  data_path = os.path.join(base_dir, "data", "student_scores.csv")
  dataset = pd.read_csv(data_path)
  # print(dataset.info())
  # rows, cols = dataset.shape
  # print(f"num of rows:{rows}, num of cols:{cols}")

  X = dataset.iloc[: , :1]
  Y = dataset.iloc[: , 1]
  # print(X.shape)
  # print(Y.head(5))
  
  # plt.scatter(X, Y)
  # plt.title("Hours vs Percentage")
  # plt.xlabel("Hours")
  # plt.ylabel("Percentage score")
  # plt.show()
  
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  sc_x = StandardScaler()
  x_train = sc_x.fit_transform(x_train)
  x_test = sc_x.transform(x_test)
  
  return x_train, x_test, y_train, y_test 

x_train, x_test, y_train, y_test = load_data()

regressor = LinearRegression()
regressor.fit(x_train, y_train)
# print("m:" , regressor.coef_)
# print("b",regressor.intercept_)
print("y = {:.2f}x + {:.2f}".format(regressor.coef_[0], regressor.intercept_))

y_pred = regressor.predict(x_test)
df = pd.DataFrame({"Actual":y_test, "predicted":y_pred})
print(df)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"mae : {MAE}, mse : {MSE} , rmse : {RMSE}")


