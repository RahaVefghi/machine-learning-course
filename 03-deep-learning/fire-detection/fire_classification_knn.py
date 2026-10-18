import cv2
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_data():
      
    data_list = []
    labels = []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "fire_dataset")
    files = glob.glob(os.path.join(dataset_dir, "*", "*"))
    
    if len(files) == 0:
      print(f"[WARNING] No dataset found in {dataset_dir}. Please place fire_dataset folder here.")
      return np.array([]), np.array([]), np.array([]), np.array([])

    for i, address in enumerate(files):
      img = cv2.imread(address)
      img = cv2.resize(img, (32,32))
      img = img/255.0
      img = img.flatten()
      # print(img.shape)
      data_list.append(img)
      label = address.split("\\")[-1].split(".")[0]
      labels.append(label)
      
      if i % 100 == 0:
        print("[INFO:]{}/{}proccessed".format(i, 1000))
      
      
    data_list = np.array(data_list)  

    x_train, x_test, y_train, y_test = train_test_split(data_list, labels, test_size=0.2) 

    print( x_train.shape)

    return x_train,x_test, y_train, y_test
  
x_train,x_test, y_train, y_test = load_data()  
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
dump(clf, "fire_detection.z")
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="fire")
recall = recall_score(y_test, y_pred, pos_label="fire")
f1_score = f1_score(y_test, y_pred, pos_label="fire")


print("precision: {:.2f}".format(precision*100))
print("recall: {:.2f}".format(recall*100))
print("f1_score: {:.2f}".format(f1_score*100))
print("accuracy: {:.2f}".format(accuracy*100))

