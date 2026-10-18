import matplotlib.pyplot as plt 
import cv2 
from sklearn.cluster import KMeans 
from sklearn.datasets import load_digits 
import numpy as np 

digits = load_digits()
# this code are for working with dataset
# img = digits.data[1300]
# img = np.reshape(img, (8,8))
# img = cv2.resize(img,(128, 128) )

# cv2.imshow("image", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)

# img = kmeans.cluster_centers_[0].reshape((8,8))
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

centers = kmeans.cluster_centers_.reshape((10,8,8))
for i in range(1,11):
  plt.subplot(2,5,i)
  plt.imshow(centers[i-1])
  
plt.show()  