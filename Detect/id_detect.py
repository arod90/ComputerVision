import cv2
import time
import os
import numpy as np
import pyrealsense2 as rs
from test.image_transform import *
#test

# Create a VideoCapture object to access the webcam
# need to add 
cap = cv2.VideoCapture(1)

# Define directory to save images
output_dir = "captured_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_count = 0
max_images = 5  # Maximum number of images to capture

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream color and depth data
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming~!
pipeline.start(config)

# Directory to save images
save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

image_count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for better performance
    ratio = frame.shape[0] / 500.0
    orig = frame.copy()
    frame = cv2.resize(frame, (int(frame.shape[1] / ratio), int(frame.shape[0] / ratio)))

    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edged = cv2.Canny(gray, 75, 200)

    # Find contours in the edged frame
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None

    # Loop over the contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If the approximated contour has four points, then it is likely the target object (ID card-like)
        if len(approx) == 4:
            screenCnt = approx
            break

    # If a four-point contour is found, apply the perspective transform
    if screenCnt is not None:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        cv2.imshow("Warped", warped)  # Show the warped ID card

    # Display the original frame with detected contours
    cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2) if screenCnt is not None else None
    cv2.imshow("Video Feed", frame)

    # Press 'q' to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()