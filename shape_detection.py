import cv2 as cv
from glob import glob
from os.path import join
import numpy as np

# Read the input image
dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\trial'
img_path = glob(join(dataset_path, '*'))

img = cv.imread(img_path[15])
image = cv.bilateralFilter(img, 5, 180, 180)
# convert the image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
'''
# apply thresholding to convert the grayscale image to a binary image
ret,binary = cv.threshold(gray,150,255,cv.THRESH_BINARY)
'''
gray = cv.blur(gray, (3,3))
#edges = cv.Canny(gray, 0, 250, apertureSize=3)
#cv.imshow('edges', edges)

#retVal, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) #value of the threshold determined automatically (bimodal distribution)
retVal, binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
#binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 3) #99 is blockSize of a pixel neighborhood to calculate a threshold value, 3 is constant subtracted from the mean or weighted sum of the neighbourhood pixels

cv.imshow('binary', binary)
# find the contours
contours,hierarchy = cv.findContours(binary, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#print("Number of contours detected:",len(contours))

# take first contour
for cnt in contours:
    area = cv.contourArea(cnt)
    # Shortlisting the regions based on their area.
    if area > 200:
        approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)# 2nd arg is epsilon value
        # Checking if the no. of sides of the selected region is 10.
        if(len(approx) > 12):
            cv.drawContours(img, [cnt], 0, (0, 0, 255), 2)
            #cv.drawContours(img, [approx], 0, (255, 0, 255), 2)
cv.imshow('detected objects', img)
cv.waitKey(0)




'''            
contour_list = []
for contour in contours:
    approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True),True)
    area = cv.contourArea(contour)
    if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
        contour_list.append(contour)

cv.drawContours(img, contour_list,  -1, (255,255,0), 2)
cv.imshow('Objects Detected',img)
cv.waitKey(0)         
'''

'''
for cnt in contours:
    hull = cv.convexHull(cnt)
    if len(hull) > 3:
        defects = cv.convexityDefects(cnt, cv.convexHull(cnt, returnPoints=False))
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv.line(img, start, end, [0, 255, 0], 2)
                cv.circle(img, far, 5, [0, 0, 255], -1)

# Showing the image along with outlined arrow.
'''



