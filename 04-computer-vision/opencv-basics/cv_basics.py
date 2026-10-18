import os

base_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_dir, "5.jpg")
img = cv2.imread(img_path)
# print(img.dtype)
# print(img.shape)
# roi = img[100:200, 300:900]
# cv2.imshow("roi image", roi)
# rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow("rgb_image",rgb_image )
# gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray_imageee", gray_image)
# blue_channel = img[:,:,0]
# cv2.imshow("blue channel", blue_channel)
# cv2.imshow("myimage", img)
# cv2.imwrite("convert_to_png.jpg",img)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

# cap = cv2.VideoCapture("MRLINE.mp4")

# while True:
#   rate, frame = cap.read()
#   if frame is None: break
#   cv2.imshow("frame", frame)
#   if cv2.waitKey(30) ==ord("q"): break
cv2.rectangle(img, (10,40), (165,458),(25,245,60), 5)
cv2.circle(img,(300,250), 50, (2,2,2), -1)
cv2.putText(img, "hello night",(150,500), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.82, (150,142,56),5)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 