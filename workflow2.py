import cv2 as cv
from glob import glob
from os.path import join
from skimage.feature import hog
import numpy as np
from skimage import exposure

dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
img_path = glob(join(dataset_path, '*'))

def find_contours(index):
    img = cv.imread(img_path[index])
    adjusted = cv.convertScaleAbs(img, alpha=1.5, beta=80)
    blurred = cv.bilateralFilter(adjusted, 20, 60, 60)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)

    #_, binary = cv.threshold(gray, 10, 255, cv.THRESH_BINARY) #threshold is 100: if less, pixel set to 0 otherwise set to 255; binary or binary inv
    #retVal, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 81, 2)
    cv.imshow('binary', binary)

    # find the contours from the thresholded image
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #mode = RETR_TREE, RETR_EXTERNAL
    cont_img = img.copy()

    for cnt in contours:
        approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True) #approximates a contour shape to another shape with less number of vertices.
        if len(approx) > 12:
            cv.drawContours(cont_img,[cnt],0,(0,0,255),3)
    cv.imshow('con_img', cont_img)
    return img, cont_img, contours, binary

def draw_circles(index):
    img, cont_img, contours, binary = find_contours(index)
    gray = cv.cvtColor(cont_img, cv.COLOR_RGB2GRAY)
    cv.imshow('gray2',gray)
    #Apply hough transform on the image
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, minDist=img.shape[0]/8, param1=100, param2=10, minRadius=20, maxRadius=30) #min dist /4 work better?
    #circles = cv.HoughCircles(fd_img, cv.HOUGH_GRADIENT, dp=1.3, minDist=30, param1=150, param2=70, minRadius=10, maxRadius=50)

    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.imshow('result img', img)
    cv.waitKey(0)
    return img

    cv.waitKey(0)

draw_circles(16)
