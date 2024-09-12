import pyrealsense2 as rs
import numpy as np
import cv2
import os


# !TODO invetigate how i can subscribe to an event in the front end, this event comes from a micro service (lambda function AWS) 
# !TODO something like an RSS feed, 
from image_transform import four_point_transform

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream color and depth data
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # Higher resolution for depth
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # Higher resolution for color

# Start streaming~!
pipeline.start(config)

# Directory to save images
save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

image_count = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap to depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))


        # Resize frame for better performance
        # ratio = color_image.shape[0] / 500.0
        # orig = color_image.copy()
        # color_image = cv2.resize(color_image, (int(color_image.shape[1] / ratio), int(color_image.shape[0] / ratio)))

        orig = color_image.copy()
        ratio = 1.0

        # Convert the color_image to grayscale and blur it
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
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
        cv2.drawContours(color_image, [screenCnt], -1, (0, 255, 0), 2) if screenCnt is not None else None

        # Display the images
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            # Press 'q' to exit the loop
            break
        elif key & 0xFF == ord('p'):
            # Press 'p' to save the current frame
            image_count += 1
            color_image_path = os.path.join(save_dir, f"color_image_{image_count}.png")
            depth_image_path = os.path.join(save_dir, f"depth_image_{image_count}.png")

            # Save color image
            cv2.imwrite(color_image_path, color_image)

            # Save depth image with colormap
            cv2.imwrite(depth_image_path, depth_colormap)

            print(f"Saved color image as {color_image_path} and depth image as {depth_image_path}")

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()