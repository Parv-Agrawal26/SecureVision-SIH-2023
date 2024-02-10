import cv2
import math
from ultralytics import YOLO
import cvzone
import pygame
import os

current_directory = os.getcwd()
file_path = os.path.join(
    current_directory, "beep.mp3")

# Initialize pygame for sound
pygame.mixer.init()
# Replace "beep.mp3" with the path to your sound file
pygame.mixer.music.load(file_path)

cap = cv2.VideoCapture(0)  # for webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-weights/yolov8n.pt")

ClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "mobile",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

previous_person_count = 0
current_person_count = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Count people detected in the current frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == ClassNames.index("person"):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Check if the person's bounding box is entirely within the frame boundaries
                if 0 <= x1 < x2 <= img.shape[1] and 0 <= y1 < y2 <= img.shape[0]:
                    current_person_count += 1
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

    # Display the count of new people detected in the current frame
    new_people_count = current_person_count - previous_person_count
    if new_people_count > 10:
        cv2.putText(img, f' People Count: {new_people_count} CROWDED AREA', (
            10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        # Play a beep sound when crowded
        pygame.mixer.music.play()

    else:
        cv2.putText(img, f' People Count: {new_people_count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update the previous count with the current count
    previous_person_count = current_person_count

    cv2.imshow("Image", img)
    cv2.waitKey(1)
