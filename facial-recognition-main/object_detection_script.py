from ultralytics import YOLO
import cv2
import cvzone
import math
import pygame
import os

current_directory = os.getcwd()
file_path = os.path.join(
    current_directory, "beep.mp3")
pygame.mixer.init()
pygame.mixer.music.load(file_path)


def main():
    cap = cv2.VideoCapture(0)  # for webcam
    cap.set(3, 1280)
    cap.set(4, 720)
    # cap = cv2.VideoCapture("Videos/bicycle.mp4") # for videos

    model = YOLO("../Yolo-weights/yolov8n.pt")

    ClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "WEAPON",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0]*100)) / 100

                # Class Name
                cls = int(box.cls[0])
                if (ClassNames[cls] == "knife"):
                    pygame.mixer.music.play()
                if (ClassNames[cls] == "WEAPON"):
                    pygame.mixer.music.play()
                    cv2.putText(img, f' there is a  {ClassNames[cls]} detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 5)
                    cvzone.putTextRect(img, f'{ClassNames[cls]} ', (max(
                        0, x1), max(35, y1)), scale=0.9, thickness=1)

                else:
                    cvzone.putTextRect(img, f'{ClassNames[cls]} ', (max(
                        0, x1), max(35, y1)), scale=0.9, thickness=1)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        # Exit the loop with a key press, e.g., the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
