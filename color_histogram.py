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
import statistics as stats


dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
img_path = glob(join(dataset_path, '*'))

#calculate color histograms of 2 images and comapres them, under assumption that images with similar color distributions contain equally similar visual contents
def calculate_hist_bgr(index):
    img = cv.imread(img_path[index])
    hist_bgr = []
    for i in range(3):
        hist_values = cv.calcHist([img], [i], None, [256], [0, 256])
        hist_bgr.append(hist_values)
        plt.plot(hist_values)
    plt.show()
    return hist_bgr


a, b = calculate_hist_bgr(15), calculate_hist_bgr(15)
brg_comparison_score = [] #comparison for each color channel
for i in range(3):
    brg_comparison_score.append(cv.compareHist(a[i], b[i], 3)) #3 is BHATTACHARYYA
if stats.mean(brg_comparison_score) < 0.3:
    print(stats.mean(brg_comparison_score), 'the bgr image might contain pears')
else:
    print(stats.mean(brg_comparison_score), 'the bgr image might not contain pears')


def calculate_hist_bw(index):
    img_bw = cv.imread(img_path[index], cv.IMREAD_GRAYSCALE)
    hist_bw = (cv.calcHist([img_bw], [0], None, [256], [0, 256]))
    plt.plot(hist_bw, color="black")
    plt.show()
    return hist_bw

x, y = calculate_hist_bw(15), calculate_hist_bw(15)
comparison = cv.compareHist(x, y, 3)
if comparison < 0.3:
    print(comparison, 'the bw image might contain pears')
else:
    print(comparison, 'the bw image might not contain pears')

