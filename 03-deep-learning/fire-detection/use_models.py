import cv2
import glob
from joblib import load
import numpy as np


import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "fire_detection.z")
test_dir = os.path.join(base_dir, "test_samples")

if not os.path.exists(model_path):
    print(f"[ERROR] Model file '{model_path}' not found. Please train the model first.")
    exit(1)

clf = load(model_path)

for item in glob.glob(os.path.join(test_dir, "*")):
    img = cv2.imread(item)

    if img is None:
        print("Cannot load:", item)
        continue

    r_img = cv2.resize(img, (32,32))
    r_img = r_img / 255.0
    r_img = r_img.flatten()
    r_img = np.array([r_img])
    label = clf.predict(r_img)[0]
    
    cv2.putText(img, label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()    
