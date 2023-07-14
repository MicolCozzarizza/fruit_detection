import cv2 as cv
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import sklearn
import skimage
from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm
import numpy as np
import random as rng

dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\trial'
img_path = glob(join(dataset_path, '*'))

img = cv.imread(img_path[15])
adjusted = cv.convertScaleAbs(img, alpha=1.5, beta=40)# to adjust contrast and brightness
#image = cv.bilateralFilter(img, 20, 60, 60)
cv.imshow('adjusted', adjusted)

gray = cv.cvtColor(adjusted, cv.COLOR_RGB2GRAY)
gray = cv.blur(gray, (3,3))
cv.imshow('gray', gray)

threshold = 80
_, binary = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
#binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 135, 2)
#canny_output = cv.Canny(gray, threshold, threshold * 2)
cv.imshow('edges', binary)


contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Find the rotated rectangles and ellipses for each contour
minRect = [None]*len(contours)
minEllipse = [None]*len(contours)
for i, c in enumerate(contours):
    minRect[i] = cv.minAreaRect(c)
    if c.shape[0] > 50:
        minEllipse[i] = cv.fitEllipse(c)

# Draw contours + rotated rects + ellipses
drawing = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

for i, c in enumerate(contours):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    # contour
    cv.drawContours(drawing, contours, i, color)
    # ellipse
    if c.shape[0] > 50:
        cv.ellipse(drawing, minEllipse[i], color, 2)
    # rotated rectangle
    box = cv.boxPoints(minRect[i])
    box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    cv.drawContours(drawing, [box], 0, color)

cv.imshow('contours',drawing)
cv.waitKey(0)