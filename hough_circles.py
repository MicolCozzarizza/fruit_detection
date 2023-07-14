import cv2 as cv
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import sklearn
import skimage
import numpy as np

dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\trial'
img_path = glob(join(dataset_path, '*'))

def draw_circles(index):
    img = cv.imread(img_path[index])
    #img = calculate_hog(index)
    #adjusted = cv.convertScaleAbs(img, alpha=1, beta=50)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.blur(gray, (3, 3))
    #gray = cv.bilateralFilter(gray, 10, 50, 50)
    cv.imshow('gray', gray)

    #Apply hough transform on the image
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, minDist=gray.shape[0]/4, param1=100, param2=10, minRadius=20, maxRadius=40) #1 is inverse ratio of the accumulator resolution to the image resolution (=1 so accumulator has the same resolution as the input image), gray.shape[0] is minDist between the centers of the detected circles. P2 is the accumulator threshold for the candidate detected circles. By increasing this threshold value, we can ensure that only the best circles, corresponding to larger accumulator values, are returned
    #circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1.3, minDist=30, param1=150, param2=70, minRadius=0, maxRadius=0)
    print(gray.shape[0]/8)
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv.circle(img, (i[0], i[1]), i[2], (255, 255, 0), 2)
            # Draw inner circle
            #cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv.imshow('result img', img)
    cv.waitKey(0)
    return gray

draw_circles(29)



