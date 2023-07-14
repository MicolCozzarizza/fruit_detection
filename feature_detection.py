import cv2 as cv
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import numpy as np


def feature_detection(index1,index2,type):
    dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
    img_path = glob(join(dataset_path, '*'))

    img_bw1 = cv.imread(img_path[index1], cv.IMREAD_GRAYSCALE)
    img_bw2 = cv.imread(img_path[index2], cv.IMREAD_GRAYSCALE)
    #print(img_path[15])

    if type == 'sift':
        sift = cv.SIFT_create() #to cretae sift object
        kp1, des1 = sift.detectAndCompute(img_bw1,None)
        kp2, des2 = sift.detectAndCompute(img_bw2,None)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    elif type == 'orb':
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_bw1, None)
        kp2, des2 = orb.detectAndCompute(img_bw2, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    img1 = cv.drawKeypoints(img_bw1, kp1, None)
    img2 = cv.drawKeypoints(img_bw2, kp2, None)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], img2, flags=2)
    #plt.imshow(img3),plt.show()
    return cv.imshow('matching image', img3), cv.waitKey(0)

feature_detection(1, 15, 'sift')




