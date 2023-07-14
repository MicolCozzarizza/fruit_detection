import cv2 as cv
from glob import glob
from os.path import join
import numpy as np

dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
img_path = glob(join(dataset_path, '*'))
img = cv.imread(img_path[32])

bilateral_filtered_image = cv.bilateralFilter(img, 5, 175, 175)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gray = cv.medianBlur(gray, 5)

# Create a SimpleBlobDetector object
detector = cv.SimpleBlobDetector_create()

# Detect blobs in the image
keypoints = detector.detect(gray)

# Draw circles on the original image
for keypoint in keypoints:
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    r = int(keypoint.size / 2)
    cv.circle(img, (x, y), r, (0, 255, 0), 2)

# Display the image
cv.imshow("Detected Circles", img)
cv.waitKey(0)
