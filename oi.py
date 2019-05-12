from kerasyolo3.yolo import YOLO
import cv2
from PIL import Image
import numpy as np

yolo_model = YOLO()
img1 = cv2.imread("videos/im100.jpg")
img2 = Image.fromarray(img1)
img2 = yolo_model.detect_image(img2)
img2 = np.asarray(img2)
while(True):
    cv2.imshow("mehdi", img1)
    cv2.imshow("yolo", img2)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()