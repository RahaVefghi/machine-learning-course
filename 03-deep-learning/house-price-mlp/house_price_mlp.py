import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

from keras.utils import to_categorical
from keras import models, layers

EPOCHS = 200
BATCH_SIZE = 32
def load_house_attributes(inputpath):
  cols = ["bedrooms", "bathroom", "area", "zipcode", "price"]
  df = pd.read_csv(inputpath, sep=" ", header=None, names=cols)
  zipcodes = df["zipcode"].value_counts().keys().to_list()
  counts = df["zipcode"].value_counts().to_list()
  for (zipcode, count) in zip(zipcodes, counts):
    
      if count < 25:
        idxs = df[df["zipcode"] == zipcode].index
        
        df.drop(idxs, inplace = True)
    
  return df

def preprocess_house_attribute(train, test):
  continous = ["bedrooms", "bathroom", "area"]
  sc = StandardScaler()
  trainContinous = sc.fit_transform(train[continous])
  testContinous = sc.fit_transform(test[continous])
  
  # encoder = OneHotEncoder(sparse_output =False, handle_unknown='ignore')
  encoder = LabelBinarizer()
  trainCtegorical = encoder.fit_transform(train["zipcode"])
  testCtegorical = encoder.fit_transform(test["zipcode"])
  trainX = np.hstack([trainContinous, trainCtegorical])
  testX = np.hstack([testContinous, testCtegorical])
  # print(trainContinous.shape)
  # print(testContinous.shape) 
  # print(trainX.shape)
  # print(testX.shape)
  # print(testCtegorical.shape)
  # print(trainCtegorical.shape)
  return trainX, testX

def neural_network(x_train, y_train, x_test, y_test):
    net = models.Sequential([
                            #layers.Flatten( input_shape=(32,32,3)),
                            layers.Dense(30, activation="relu"),
                            layers.Dense(8, activation="relu"),
                            layers.Dense(1, activation="linear")
                            
                              ])
    net.summary()

    net.compile(optimizer="SGD",
                loss="mse")

    H = net.fit(x_train, y_train, BATCH_SIZE, EPOCHS, validation_data=(x_test, y_test))
    
    loss = net.evaluate(x_test, y_test)
    print("loss:{:.2f}".format(loss))
    
    net.save("mlp.h5")
      
    return net
  
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "HousesInfo.txt")
df = load_house_attributes(data_path)  
train, test = train_test_split(df, test_size=0.2, random_state=42)  
trainX, testX = preprocess_house_attribute(train, test)  

maxPrice = train["price"] .max() 
trainY = train["price"]/maxPrice  
testY = test["price"]/maxPrice  

model= neural_network(trainX, trainY,testX, testY)

preds = model.predict(testX)
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
abspercentDiff = np.abs(percentDiff)
mean = np.mean(abspercentDiff)
std = np.std(abspercentDiff)
print("[INFO] mean: {:.2f}%, std: {:.2f}".format(mean, std))
# mape = mean_absolute_percentage_error(testY, preds)
# print("mape:{:.2f}".format(mape*100))
