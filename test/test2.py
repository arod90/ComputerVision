import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream color and depth data
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
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

        # Display the images
        cv2.imshow('RealSense', images)

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