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


#dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
img_path = glob(join(dataset_path, '*'))


#HOG
def calculate_hog(index):
    img = cv.imread(img_path[index])#   img = cv.imread(img_path[index], cv.IMREAD_GRAYSCALE)

    #img = cv.resize(img, (128*4, 64*4))
    img = cv.resize(img, (64*2, 128*2))
    cv.imshow('image', img)
    #img = canny_edges(index)
    fd, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, visualize=True, channel_axis = -1)#fd is feature matrix,
    #fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
    #ax.imshow(hog_img, cmap=plt.cm.gray)
    #plt.show()
    #print(fd)
    cv.imshow('Hog image', hog_img)
    cv.waitKey(0)
    #print(fd)
    return fd, hog_img

calculate_hog(27)
