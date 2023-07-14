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

dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
img_path = glob(join(dataset_path, '*'))

def calculate_hog(index):
    img = cv.imread(img_path[index])
    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    #img = cv.resize(img, (64*3, 128*3))
    fd, fd_img = hog(lab, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, visualize=True, channel_axis= -1)#fd is feature matrix,

    cv.imshow('Hog image', fd_img)
    cv.waitKey(0)
    return img, fd, fd_img
calculate_hog(32)