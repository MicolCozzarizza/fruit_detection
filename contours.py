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

dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
img_path = glob(join(dataset_path, '*'))


def draw_contours(index):
    img = cv.imread(img_path[index])
    #fd, img = calculate_hog(index)
    adjusted = cv.convertScaleAbs(img, alpha=1, beta=50)# to adjust contrast and brightness
    image = cv.bilateralFilter(adjusted, 20, 60, 60) #for smoothening images, reducing noise but preserving edges. 15 is diameter of pixel neighborhood, 80 is the filter sigma in the color space. 80 is filter sigma in the coordinate space.
    #image = cv.blur(image, (3,3))
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    #gray = cv.equalizeHist(gray) #to improve contrast
    cv.imshow('gray blurred', gray)

    #get binary image either with canny edges or with threshold
    #gray = cv.normalize(gray, None, alpha=0,beta=200, norm_type=cv.NORM_MINMAX)
    gray = cv.Canny(image, 100, 200, L2gradient= True)
    cv.imshow('gray', gray)
    # create a binary thresholded image
    #_, binary = cv.threshold(gray, 10, 255, cv.THRESH_BINARY) #threshold is 100: if less, pixel set to 0 otherwise set to 255; binary or binary inv
    #retVal, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 81, 2)

    cv.imshow('binary img', gray)

    # find the contours from the thresholded image+6
    contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #mode = RETR_TREE, RETR_EXTERNAL

    for cnt in contours:
        approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True) #approximates a contour shape to another shape with less number of vertices.
        if len(approx) > 12:
            cv.drawContours(img,[cnt],0,(0,0,255),2)
            #cv.drawContours(img, [approx], -1, (0,255,255), 1)
    cv.imshow('image approx contours', img)

    image2 = cv.drawContours(img, contours, -1, (0, 0, 255), 2)
    cv.imshow('image', image2)

    cv.waitKey(0)
    return tuple(contours), image, gray

draw_contours(23)
