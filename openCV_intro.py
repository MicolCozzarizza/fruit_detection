import cv2 as cv
from glob import glob
from os.path import join


dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
img_path = glob(join(dataset_path, '*'))

#for i in range(len(img_path):

img = cv.imread(img_path[2]) #[i]
#cv.imshow('pear image', img)
#cv2.waitKey(0)

img_size = img.shape
#draw circle

center_coordinates = (img_size[1]//2, img_size[0]//2) #x , y coordinates
radius = 20
circle = cv.circle(img, center_coordinates, radius=20, color=(0, 0, 0), thickness=2)
cv.imshow(f'{img_path}', img)
cv.waitKey(0)

img_bw = cv.imread(img_path[1], cv.IMREAD_GRAYSCALE)
contour = cv.imcontour















