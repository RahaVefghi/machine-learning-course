import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import models, layers

EPOCHS = 20
BATCH_SIZE = 32

def load_data():
      
    data_list = []
    labels = []
    le = LabelEncoder()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "fire_dataset")
    files = glob.glob(os.path.join(dataset_dir, "*", "*"))

    if len(files) == 0:
      print(f"[WARNING] No dataset found in {dataset_dir}. Please place fire_dataset folder here.")
      return np.array([]), np.array([]), np.array([]), np.array([]), 0, None

    for i, address in enumerate(files):
      img = cv2.imread(address)
      img = cv2.resize(img, (32,32))
      img = img/255.0
      # img = img.flatten()
      # print(img.shape)
      data_list.append(img)
      label = address.split("\\")[-1].split(".")[0]
      labels.append(label)
      
      if i % 100  == 0:
        print("[INFO:]{}/{}proccessed".format(i, 1000))
      
      
    data_list = np.array(data_list)  

    x_train, x_test, y_train, y_test = train_test_split(data_list, labels, test_size=0.2,random_state=42)
    
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test )
    
    print(y_train[5])
    

  
    return x_train,x_test, y_train, y_test
  
def neural_network(x_train, y_train, x_test, y_test):
    net = models.Sequential([
                            layers.Flatten( input_shape=(32,32,3)),
                            layers.Dense(20, activation="relu"),
                            layers.Dense(8, activation="relu"),
                            layers.Dense(2, activation="softmax")
                            
                              ])
    net.summary()

    net.compile(optimizer="SGD",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    H = net.fit(x_train, y_train, BATCH_SIZE, EPOCHS, validation_data=(x_test, y_test))
    
    loss, acc = net.evaluate(x_test, y_test)
    print("loss:{:.2f}, accuracy:{:.2f}".format(loss, acc))
    
    net.save("mlp.h5")
      
    return H 

def show_result(H):
        
    # print(x_train.shape)
    # print(x_test.shape)
    print(H.history.keys())
    plt.style.use("ggplot") 
    plt.plot(np.arange(EPOCHS), H.history["loss"], label = "train loss")
    plt.plot(np.arange(EPOCHS), H.history["val_loss"], label = "test loss")
    plt.plot(np.arange(EPOCHS), H.history["accuracy"], label = "train accuracy")
    plt.plot(np.arange(EPOCHS), H.history["val_accuracy"], label = "test accuracy")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("training on fire dataset")
    plt.show()

x_train,x_test, y_train, y_test = load_data()  
H = neural_network(x_train, y_train, x_test, y_test)
show_result(H)
 