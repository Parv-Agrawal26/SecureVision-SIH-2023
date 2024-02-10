import cv2
import os
import time

name = 'Criminal'  # Replace with your name
output_folder = "dataset/" + name + "/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

cam = cv2.VideoCapture(0)

cv2.namedWindow("Automatic Photo Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Automatic Photo Capture", 500, 300)

img_counter = 0
max_images = 50
capture_interval = 0.5 # Capture an image every 5 seconds (adjust as needed)

while img_counter < max_images:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Automatic Photo Capture", frame)

    if img_counter == 0 or time.time() - start_time >= capture_interval:
        img_name = output_folder + "image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        start_time = time.time()

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()
