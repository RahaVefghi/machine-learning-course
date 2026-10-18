import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import glob
import numpy as np
from tensorflow.keras.models import load_model

EPOCHS = 20
BATCH_SIZE = 32
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "mlp.h5")
test_dir = os.path.join(base_dir, "test_samples")

if not os.path.exists(model_path):
    print(f"[ERROR] Model file '{model_path}' not found. Please train the model first.")
    exit(1)

clf = load_model(model_path)
output_label = ["fire", "non_fire"]

for item in glob.glob(os.path.join(test_dir, "*")):
  
  img =cv2.imread(item)
  if img is None:
    continue
  
  r_image = cv2.resize(img, (32,32))
  r_image = r_image/255.0 
  r_image = np.expand_dims(r_image, axis=0)
  
  pred_output = clf.predict(r_image)[0]
  max_pred = np.argmax(pred_output)
  
  output = output_label[max_pred]
  
  if output == "fire":
    cv2.putText(img, "Fire!!!!", (10,20),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.95,(0,0,255,3))
  else:
    cv2.putText(img, "None fire", (10,20),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.95,(0,0,255,3))  
 
  cv2.imshow("image", img)
  cv2.waitKey(0)
cv2.destroyAllWindows()  
 