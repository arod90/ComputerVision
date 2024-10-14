import random
import cv2
from services.AWS_Service import S3Uploader
from services.mb_chalice_service import MBChaliceService

def get_hsv_values(image_bgr):
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Click event function
    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, s, v = hsv_image[y, x]
            print(f'HSV at ({x}, {y}): H={h}, S={s}, V={v}')
    
    cv2.namedWindow('Original Image')
    cv2.setMouseCallback('Original Image', mouse_event)
    
    while True:
        cv2.imshow('Original Image', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def detect_card_color(card_image):
    """
    Determines if the card_image has a majority green or blue tint.
    Displays intermediate images for analysis.
    Returns 'green' or 'blue' based on the dominant color.
    """
    import cv2
    import numpy as np


    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)

    # Display the HSV image (convert back to BGR for proper color display)
    hsv_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV Image', hsv_bgr)
    cv2.waitKey(1)

    # # Assuming card_image is your cropped card image
    # hsv_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)

    # # Flatten the HSV image array and convert to a list
    # h_values = hsv_image[:, :, 0].flatten()
    # s_values = hsv_image[:, :, 1].flatten()
    # v_values = hsv_image[:, :, 2].flatten()

    # # Print the average HSV values
    # print(f"Average H: {np.mean(h_values)}")
    # print(f"Average S: {np.mean(s_values)}")
    # print(f"Average V: {np.mean(v_values)}")

    # Define HSV ranges for green color
    lower_green = np.array([20, 15, 190])
    upper_green = np.array([40, 50, 220])

    # Define HSV ranges for blue color
    lower_blue = np.array([95, 40, 40])    # H:95-135, S:40-255, V:40-255
    upper_blue = np.array([135, 255, 255])

    # Define HSV ranges for white color (low saturation, high value)
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([179, 20, 255])

    # Create masks for green, blue, and white pixels
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    # Display the masks
    cv2.imshow('Green Mask', mask_green)
    cv2.imshow('Blue Mask', mask_blue)
    cv2.imshow('White Mask', mask_white)
    cv2.waitKey(1)

    # Count the number of pixels in each mask
    num_green = cv2.countNonZero(mask_green)
    num_blue = cv2.countNonZero(mask_blue)
    num_white = cv2.countNonZero(mask_white)
    num_blue_white = num_blue + num_white

    # Determine which tint is more prevalent
    if num_green > num_blue_white:
        result = "old"
    elif num_green < num_blue_white:
        result = "new"
    else:
        result = "none"

    # Return the dominant color
    return result

def upload_and_process_to_s3(card_type, card_image):
    # Initialize S3 uploader and MBChaliceService
    s3_uploader = S3Uploader(bucket_name="mb-id-storage", region_name='us-east-2')
    mbService = MBChaliceService()

    object_url = None
    # Perform validations based on card color
    if card_type == 'old':
        # Process for old cards
        image_filename = f'green_card_{random.randrange(0,100000)}.jpg'
        cv2.imwrite(image_filename, card_image)
        object_url = s3_uploader.upload_cv2_image(card_image, f"mb-test-aa/front/old/{image_filename}")
        print(f"Old card saved as {image_filename}")
    elif card_type == 'new':
        # Process for new cards
        image_filename = f'blue_card_{random.randrange(0,100000)}.jpg'
        cv2.imwrite(image_filename, card_image)
        object_url = s3_uploader.upload_cv2_image(card_image, f"mb-test-aa/front/new/{image_filename}")
        print(f"New card saved as {image_filename}")
    else:
        # Handle unexpected cases
        print("Unknown card type detected.")
        return None

    # Process image
    response = mbService.post(data={"id_type": card_type, "front_image_url": object_url})
    
    # Convert the random integer to a string for MessageDeduplicationId
    deduplication_id = str(random.randrange(0,100000))
    s3_uploader.notify_image_processed(deduplication_id, object_url)
    
    s3_url = s3_uploader.poll_sqs_fifo()
    return object_url