import cv2
import numpy as np
import time
from services.mb_chalice_service import MBChaliceService
from services.AWS_Service import S3Uploader

# initialize s3 uploader
s3_uploader = S3Uploader(bucket_name="mb-id-storage", region_name='us-east-2')
mbService = MBChaliceService()

# Initialize video capture
cap = cv2.VideoCapture(1)

# Set camera resolution to high values (adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

card_saved = False
card_detected_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy of the original frame for saving the image later
    original_frame = frame.copy()

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
                    if elapsed_time >= 2 and not card_saved:
                        # Get the bounding rect
                        x, y, w, h = cv2.boundingRect(approx)

                        # Ensure coordinates are within frame bounds
                        x = max(x, 0)
                        y = max(y, 0)
                        w = min(w, original_frame.shape[1] - x)
                        h = min(h, original_frame.shape[0] - y)

                        # Crop the card from the original frame (without contour lines)
                        card_image = original_frame[y:y+h, x:x+w]

                        # Save the image
                        # cv2.imwrite('card_image.jpg', card_image)
                        

                        object_url = s3_uploader.upload_cv2_image(card_image, "mb-test-aa/front/front-captured.jpg")
                        
                        s3_uploader.notify_image_processed("TestID5", object_url)


                        s3_url = s3_uploader.poll_sqs_fifo()
                        response = mbService.post(data={"front_image_url": s3_url})
                        card_saved = True
                        
                        # print(response)

                # Draw the contour on the frame for visualization
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

                break  # Stop processing contours after finding the card

    if not card_present:
        # Reset the timer and card_saved flag if the card is not present
        card_detected_time = None
        card_saved = False

    # Show the frame with contours
    cv2.imshow('Frame', frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
