import cv2 as cv
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import sklearn
import skimage
from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm
#from hog import calculate_hog
import numpy as np


dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'

img_path = glob(join(dataset_path, '*'))
img_bw = cv.imread(img_path[11], cv.IMREAD_GRAYSCALE)

#fd, img_bw = calculate_hog(15)

hist_bw = (cv.calcHist([np.uint8(img_bw)], [0], None, [256], [0, 256]))
plt.plot(hist_bw, color="black")
plt.show()
#print(hist_bw)


#analysis of all the image pixel values to mark those that could correspond to the color green of the pears
height, width = img_bw.shape
for width_value in range(width):
    for height_value in range(height):
        if img_bw[height_value, width_value] == 20:
            cv.circle(img_bw, (width_value, height_value), 3, (255, 255, 0), -1)

cv.imshow('img', img_bw)
cv.waitKey(0)



'''
for value in hist_bw:
    if 0<=value<=0.3:
        cv.circle()

cv.imshow('img', img_bw)


img_bw = canny_edges(15)
hist_bw = (cv.calcHist([np.uint8(img_bw)], [0], None, [256], [0, 256]))
plt.plot(hist_bw, color="black")
plt.show()
'''







