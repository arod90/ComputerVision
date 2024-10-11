# this one does not have the intelisense camera
import cv2
import numpy as np
import time
import random
from services.mb_chalice_service import MBChaliceService
from services.AWS_Service import S3Uploader

# Initialize S3 uploader and MBChaliceService (commented out since implementations are not provided)
s3_uploader = S3Uploader(bucket_name="mb-id-storage", region_name='us-east-2')
mbService = MBChaliceService()

# Initialize video capture
cap = cv2.VideoCapture(1)

# Set camera resolution to high values (adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

card_saved = False
card_detected_time = None
lookup_paused = False
pause_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy of the original frame for saving the image later
    original_frame = frame.copy()

    if lookup_paused:
        # Check if 5 seconds have passed since the pause started
        elapsed_pause_time = time.time() - pause_start_time
        if elapsed_pause_time >= 4:
            # Unpause the lookup process
            lookup_paused = False
            pause_start_time = None
            # Reset flags
            card_saved = False
            card_detected_time = None
            print("Lookup process restarted.")
        else:
            # Display a message on the frame indicating the pause
            remaining_time = max(0, 5 - elapsed_pause_time)
            cv2.putText(frame, f"Processing... {int(remaining_time)}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # Show the frame
            cv2.imshow('Frame', frame)
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
    else:
        # Process frame to detect card
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
                        if elapsed_time >= 1 and not card_saved:
                            # Get the bounding rect
                            x, y, w, h = cv2.boundingRect(approx)

                            # Ensure coordinates are within frame bounds
                            x = max(x, 0)
                            y = max(y, 0)
                            w = min(w, original_frame.shape[1] - x)
                            h = min(h, original_frame.shape[0] - y)

                            # Crop the card from the original frame (without contour lines)
                            card_image = original_frame[y:y+h, x:x+w]

                            # Save the image with a unique filename
                            image_filename = f'card_image_{random.randrange(0,100000)}.jpg'
                            cv2.imwrite(image_filename, card_image)
                            print(f"Card Saved as {image_filename}")

                            # Optional: Upload the image and process it (commented out)
                            object_url = s3_uploader.upload_cv2_image(card_image, "mb-test-aa/front/back-captured_1714.jpg")
                            response = mbService.post(data={"front_image_url": object_url})
                            s3_uploader.notify_image_processed(random.randrange(0,100000), object_url)
                            s3_url = s3_uploader.poll_sqs_fifo()

                            # Provide a visual signal on the frame
                            cv2.putText(frame, "Image Captured", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # Pause the lookup process
                            lookup_paused = True
                            pause_start_time = time.time()
                            card_saved = True
                            print("Lookup process paused for 5 seconds.")

                    # Draw the contour on the frame for visualization
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    break  # Stop processing contours after finding the card

        if not card_present:
            # Reset the timer and card_saved flag if the card is not present
            card_detected_time = None
            card_saved = False

        # Display a countdown timer for capturing
        if card_detected_time is not None and not card_saved:
            elapsed_time = time.time() - card_detected_time
            remaining_time = max(0, 2 - elapsed_time)
            cv2.putText(frame, f"Capturing in {remaining_time:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with contours and messages
        cv2.imshow('Frame', frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
