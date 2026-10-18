import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow
from mtcnn import MTCNN
import cv2

detector = MTCNN()

base_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_dir, "maryam.jpg")
img = cv2.imread(img_path)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = detector.detect_faces(rgb_img)[0]
x, y, w, h = out["box"]
confidence = out["confidence"]

  
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2 )
text = "prob:{:.2f}".format(confidence*100)
cv2.putText(img, text, (x, y-20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255, 58), 2)  

kp = out["keypoints"]

for key, value in kp.items():
  cv2.circle(img, value, 8, (0, 0, 255), -1)
 
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 