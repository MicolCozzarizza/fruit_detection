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
from PIL import Image

#dataset_path = r'C:\Users\mcozzarizza\Desktop\output_imgs'
dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
#dataset_path = r'C:\Users\mcozzarizza\Desktop\trial'
img_path = glob(join(dataset_path, '*'))
def negative_img(index):
    img = cv.imread(img_path[index])
    #image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    colored_negative = abs(255-img)
    #cv.imshow('negative', colored_negative)
    #cv.waitKey(0)
    return colored_negative

x = negative_img(15)

img = cv.imread(img_path[15])
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB) #to remap in L*a*b colorspace
cv.imshow('lab img', lab)
cv.waitKey(0)

#img_path1 = r'C:\Users\mcozzarizza\Desktop\result img_screenshot_24.04.2023_2.png'
#img_path2 = r'C:\Users\mcozzarizza\Desktop\result img_screenshot_24.04.2023_neg2.png'
'''
img1 = cv.imread(img_path[33])
img2 = cv.imread(img_path[31])
height, width, channel = img1.shape
white = np.ones((height, width, 3), dtype = np.uint8)*255

addition = cv.add(img1, white)
cv.imshow('adding img', addition)

blended = cv.addWeighted(img1, 0.5, img2, 0.5, 0.0)

cv.imshow('original', img1)
cv.imshow('blended img', blended)
cv.waitKey(0)

'''




