import cv2 as cv
from glob import glob
from os.path import join
import numpy as np

#dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
dataset_path = r'C:\Users\mcozzarizza\Desktop\trial'
img_path = glob(join(dataset_path, '*'))
img = cv.imread(img_path[12])

#bilateral_filtered_image = cv.bilateralFilter(img, 5, 175, 175)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gray = cv.medianBlur(gray, 5)


cv.imshow('gray', gray)

# create a binary thresholded image
#_, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY) #threshold is 100: if less, pixel set to 0 otherwise set to 255; binary or binary inv
#_, binary = cv.threshold(gray, 245, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 99, 3)
cv.imshow('binary', binary)

# Finding contours for the thresholded image
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True) #approximates a contour shape to another shape with less number of vertices.
    if len(approx) > 12:
        cv.circle(img,center,radius,(0,255,0),2)

cv.imshow('detected circles',img)
print('n contours:', len(contours))

# create hull array for convex hull points
hull = []

# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv.convexHull(contours[i], False))

#draw convex hull
# create an empty black image
drawing = np.zeros((binary.shape[0], binary.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (0, 0, 255) # blue - color for convex hull
    # draw ith contour
    cv.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv.drawContours(drawing, hull, i, color, 1, 8)

cv.imshow('contour and hull pts', drawing)
cv.waitKey(0)