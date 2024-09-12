from test.image_transform import *
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
# from PIL import Image
# import pytesseract
import cv2
import os

configs = {
    "img_name": "IMG_2447",
    "preprocess": "thresh"
}
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(f"/Users/aaronsalazar/Documents/Developer/Aaron/ID_QR_Reader/images2/{configs['img_name']}.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: detect edges")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) 
print("STEP 2: Find contours")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
warped_color = warped.copy()
# cv2.imwrite("WARPED_COLOR.png", warped_color)
# cv2.imshow("WARPED", warped_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
name = "scanned_{}.png".format(configs["img_name"])
cv2.imwrite(name, warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# use color image
warped_color = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
# read the text in the scanned version of the image
# check to see if we should apply thresholding to preprocess the image
if configs["preprocess"] == "thresh":
	warped_color = cv2.threshold(warped_color, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
elif configs["preprocess"] == "blur":
	warped_color = cv2.medianBlur(warped_color, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "scanned_{}.png".format(configs["img_name"])
cv2.imwrite(filename, warped_color)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
# pickle = Image.open(filename)
# text = pytesseract.image_to_string(pickle)
# os.remove(filename)
# print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output Warped", warped_color)
cv2.waitKey(0)
cv2.destroyAllWindows()