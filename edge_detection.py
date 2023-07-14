import cv2 as cv
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import numpy as np


dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\trial'
img_path = glob(join(dataset_path, '*'))

#EDGE DETECTION
def canny_edges(index):
    img = cv.imread(img_path[index])
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    #img = cv.blur(img, (3,3))
    #img = cv.bilateralFilter(img, 15, 80, 80) #(img, 5, 190, 190)

    edged = cv.Canny(img, 100, 200, L2gradient= True) #smallest value for edge linking; largest value to find initial segments of strong edges
    cv.imshow('Canny edges', edged)
    cv.imshow('image', img)
    cv.waitKey(0)

    return edged

canny_edges(32)


'''
median_value = np.median(gray)
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * median_value))
upper = int(min(255, (1.0 + sigma) * median_value))
edged = cv.Canny(gray, lower, upper)
'''