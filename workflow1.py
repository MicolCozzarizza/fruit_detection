import cv2 as cv
from glob import glob
from os.path import join
from skimage.feature import hog
import numpy as np
from skimage import exposure

dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
img_path = glob(join(dataset_path, '*'))

def calculate_hog(index):
    img = cv.imread(img_path[index])
    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    #img = cv.resize(img, (64*3, 128*3))
    fd, fd_img = hog(lab, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, visualize=True, channel_axis= -1)#fd is feature matrix,

    return img, fd, fd_img

def canny_edges(index):
    img = cv.imread(img_path[index])
    #img = cv.blur(img, (2,2)) #optional
    blur = cv.bilateralFilter(img, 20, 60, 60) #optional
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    fd_img = cv.Canny(gray, 100, 200, L2gradient=True)
    return img, fd_img

def draw_circles(index, descriptor_type):
    if descriptor_type == 'hog':
        img, fd, fd_img = calculate_hog(index)
        fd_img = exposure.rescale_intensity(fd_img, out_range=(0, 255))
        fd_img = fd_img.astype("uint8")
    elif descriptor_type =='canny':
        img, fd_img = canny_edges(index)
    cv.imshow('fd img', fd_img)

    #Apply hough transform on the image
    circles = cv.HoughCircles(fd_img, cv.HOUGH_GRADIENT, 1, minDist=fd_img.shape[0]/8, param1=100, param2=10, minRadius=25, maxRadius=35)
    #circles = cv.HoughCircles(fd_img, cv.HOUGH_GRADIENT, dp=1.3, minDist=30, param1=150, param2=70, minRadius=10, maxRadius=50)
    #circles = cv.HoughCircles(fd_img, cv.HOUGH_GRADIENT, 1.5, minDist=fd_img.shape[0]/20, param1=50, param2=20, minRadius=fd_img.shape[0]//30, maxRadius=fd_img.shape[0]//15)

    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.imshow('result img', img)
    cv.waitKey(0)
    return img

draw_circles(15, 'canny')
