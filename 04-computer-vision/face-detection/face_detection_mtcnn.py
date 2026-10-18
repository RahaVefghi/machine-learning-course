import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
from mtcnn import MTCNN

def detect_and_draw_face():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "maryam.jpg")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not load image from {img_path}")
        return

    detector = MTCNN()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_img)

    if len(results) == 0:
        print("[INFO] No faces detected in the image.")
        return

    for face in results:
        x, y, w, h = face["box"]
        confidence = face["confidence"]

        # Draw green bounding box around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, f"Face: {confidence * 100:.1f}%",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # Draw red facial landmark points (eyes, nose, mouth corners)
        for point in face["keypoints"].values():
            cv2.circle(img, point, 4, (0, 0, 255), -1)

    print(f"[INFO] Successfully detected {len(results)} face(s).")
    cv2.imshow("MTCNN Face & Landmark Detection", img)
    print("Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_draw_face()