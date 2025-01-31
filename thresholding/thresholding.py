#pylint:disable=no-member

#### CONVERT AN IMAGE TO BINARY IMGAE - BLACK AND WHITE ####################

import cv2 as cv
import numpy as np

img = cv.imread('opencv_basics/resources/Photos/cats.jpg')
cv.imshow('img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)




## simple thresholding - manually specify your treshold value
threshold, thresh_img = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) # sets values greater than 150 to 255
cv.imshow("simple treshold", thresh_img)

threshold, thresh_img_inv = cv.threshold(gray, thresh=150, maxval=255, type=cv.THRESH_BINARY_INV) # sets values less than 150 to 255
cv.imshow("simple treshold inverse", thresh_img_inv)




## adaptive threshold - let the computer set the treshold by it self
# adaptive_thresh_img = cv.adaptiveThreshold(gray,maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
#                                            thresholdType=cv.THRESH_BINARY, blockSize=15, C=0)

adaptive_thresh_img = cv.adaptiveThreshold(gray,maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           thresholdType=cv.THRESH_BINARY_INV, blockSize=21, C=3)

cv.imshow("adaptive thresholding", adaptive_thresh_img)




cv.waitKey(0)