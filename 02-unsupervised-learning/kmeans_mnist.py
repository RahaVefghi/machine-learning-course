import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import numpy as np 
import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "mnist.npz")

if os.path.exists(data_path):
    data = np.load(data_path)
    X_train = data["x_train"]
    Y_train = data["y_train"]
    X_test = data["x_test"]
    y_test = data["y_test"]
else:
    (X_train, Y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# img = X_test[100]
# img = np.reshape(img, (28,28))
# img = cv2.resize(img, (256,256))
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

X_train = X_train/255.0
X_train = X_train.reshape((-1, 784))
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(X_train)

center= kmeans.cluster_centers_.reshape((10, 28, 28))
for i in range(10):
  plt.subplot(2,5, i+1)
  plt.imshow(center[i], cmap="gray")
plt.show()  