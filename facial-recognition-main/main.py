from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("I/3.jpg", show=True)
cv2.waitKey(0)
