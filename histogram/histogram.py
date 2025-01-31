#pylint:disable=no-member

#### ANALYSING DISTRIBUTION OF PIXEL INTENSISTY ####################

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('opencv_basics/resources/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('Mask', masked)

# # GRayscale histogram
# # hitSize which is also the bin shows the number of pixel intensity - 0-255 for grayscale 256 in number
# gray_hist = cv.calcHist(images=[gray], channels=[0], mask=mask, histSize=[256], ranges=[0,256])

# plt.figure()
# plt.title("Gray Hist")
# plt.xlabel("bins")
# plt.xlim([0,256])
# plt.ylabel("# of pixels")
# plt.plot(gray_hist)

# plt.show()


# colored histogram

plt.figure()
plt.title("Colored Hist")
plt.xlabel("bins")
plt.xlim([0,256])
plt.ylabel("# of pixels")

colors = ('b','g','r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, color=col)

plt.show()

cv.waitKey(0)