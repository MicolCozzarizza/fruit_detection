
from histogram import img_path, svm, cv
import cv2 as cv
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import sklearn
from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def calculate_features(reference_set):
    labels = []
    hog_hist = []
    for image_path in reference_set:
        if 'pear' in image_path:
            labels.append(1)
        else:
            labels.append(0)
    for i in range(len(reference_set)):
        image = cv.imread(reference_set[i], cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (128, 256))
        img_hog = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_hist.append(img_hog)
    return hog_hist, labels

dataset_path = r'C:\Users\mcozzarizza\Desktop\pear_img'
img_path = glob(join(dataset_path, '*'))

trainset = img_path[:int(0.8*len(img_path))]
testset = img_path[int(0.8*len(img_path)):]

train_x, train_y = calculate_features(trainset)
test_x, test_y = calculate_features(testset)

clf = svm.SVC(gamma=0.001, C=100., kernel='rbf', verbose=False)
#svm_model = LinearSVC(random_state=42, tol=1e-5)
clf.fit(train_x, train_y)

y_pred = clf.predict(test_x)

print(y_pred)

print('Final Accuracy: {:.3f}'.format(accuracy_score(test_y, y_pred)))


'''
clf = svm.SVC(gamma=0.001, C=100., kernel='rbf', verbose=False)
trainset = img_path[:int(0.8*len(img_path))]
testset = img_path[int(0.8*len(img_path)):]

labels = ['pear'*x for x in range(len(img_path))]
hog_images = []
for i in len(img_path):
    image = cv.imread(img_path[i])
image = cv.resize(image, (128, 256))
fd, hog_img = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
#ax.imshow(hog_img, cmap=plt.cm.gray)
hog_images.append(fd)
'''