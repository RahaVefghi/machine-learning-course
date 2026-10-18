import os
import cv2

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "5.jpg")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not load image from {img_path}")
        return

    print(f"Image Loaded: Shape = {img.shape}, Data Type = {img.dtype}")

    # Drawing geometric shapes and annotations
    # 1. Rectangle
    cv2.rectangle(img, (10, 40), (165, 458), (25, 245, 60), 5)
    # 2. Filled Circle
    cv2.circle(img, (300, 250), 50, (2, 2, 2), -1)
    # 3. Text
    cv2.putText(img, "OpenCV Demo", (150, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 142, 56), 3)

    cv2.imshow("OpenCV Basics Demo", img)
    print("Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()