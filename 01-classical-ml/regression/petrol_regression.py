import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "petrol_consumption.csv")
dataset = pd.read_csv(data_path)
X = dataset[['Petrol_tax','Average_income','Paved_Highways','Population_Driver_licens']]
Y = dataset[['Petrol_Consumption']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

sgd_regressor = SGDRegressor(max_iter=1000, random_state=40, eta0=0.001)
sgd_regressor.fit(x_train, y_train) 

df = pd.DataFrame(regressor.coef_.T, X.columns, columns=['coefficent'])
sgd_df = pd.DataFrame(sgd_regressor.coef_.T, X.columns, columns=['coefficent'])
print("numbers of epochs:", sgd_regressor.n_iter_)
print("numbers of updates:", sgd_regressor.t_)
print(x_train.shape)
print(df)
print(sgd_df)

# y_pred = regressor.predict(x_test)
# pred_df = pd.DataFrame({"actual:": list(y_test.iloc[:, 0]), "predict:": list(y_pred)})

# print("MAE:",mean_absolute_error(y_test, y_pred))
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))