import cv2
import numpy as np
import time
from utils.utilities import detect_card_color, upload_and_process_to_s3
import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()
# Configure the pipeline to stream color and depth data
config = rs.config()
# Start streaming
pipeline.start(config)

card_saved = False
card_detected_time = None
lookup_paused = False
pause_start_time = None

card_detection_time = 1
camera_pause_time = 1
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Create a copy of the original frame for saving the image later
    original_frame = np.copy(color_image)  # Changed this line

    # Rest of your code remains the same...

    if lookup_paused:
        # Check if 5 seconds have passed since the pause started
        elapsed_pause_time = time.time() - pause_start_time
        if elapsed_pause_time >= camera_pause_time:
            # Unpause the lookup process
            lookup_paused = False
            pause_start_time = None
            # Reset flags
            card_saved = False
            card_detected_time = None
            print("Lookup process restarted.")
        else:
            # Display a message on the frame indicating the pause
            remaining_time = max(0, camera_pause_time - elapsed_pause_time)
            cv2.putText(color_image, f"Processing... {int(remaining_time)}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # Show the frame
            cv2.imshow('Frame', color_image)
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
    else:
        # Process frame to detect card
        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        card_present = False

        # Loop over contours
        for cnt in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # If the approximated contour has 4 vertices, it could be our card
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > 10000:  # Adjust area threshold as needed
                    card_present = True

                    if card_detected_time is None:
                        # Start the timer
                        card_detected_time = time.time()
                    else:
                        # Check if 2 seconds have passed since the card was detected
                        elapsed_time = time.time() - card_detected_time
                        if elapsed_time >= card_detection_time and not card_saved:
                            # Get the bounding rect
                            x, y, w, h = cv2.boundingRect(approx)

                            # Ensure coordinates are within frame bounds
                            x = max(x, 0)
                            y = max(y, 0)
                            w = min(w, original_frame.shape[1] - x)
                            h = min(h, original_frame.shape[0] - y)

                            # Crop the card from the original frame (without contour lines)
                            card_image = original_frame[y:y+h, x:x+w]

                            # Determine the card color
                            card_type = detect_card_color(card_image)
                            print(f"Detected card color: {card_type}")

                            # process image and upload to s3 and notify
                            upload_and_process_to_s3(card_type, card_image)

                            # Provide a visual signal on the frame
                            cv2.putText(color_image, f"{card_type.capitalize()} Card Captured", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # Pause the lookup process
                            lookup_paused = True
                            pause_start_time = time.time()
                            card_saved = True
                            print("Lookup process paused for 5 seconds.")

                    # Draw the contour on the frame for visualization
                    cv2.drawContours(color_image, [approx], -1, (0, 255, 0), 3)
                    break  # Stop processing contours after finding the card

        if not card_present:
            # Reset the timer and card_saved flag if the card is not present
            card_detected_time = None
            card_saved = False

        # Display a countdown timer for capturing
        if card_detected_time is not None and not card_saved:
            elapsed_time = time.time() - card_detected_time
            remaining_time = max(0, card_detection_time - elapsed_time)
            cv2.putText(color_image, f"Capturing in {remaining_time:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with contours and messages
        cv2.imshow('Frame', color_image)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
pipeline.stop()
cv2.destroyAllWindows()