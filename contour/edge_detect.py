############ GRDIENT AND EDGE-DETECTION apart fro canny method ####################

import cv2 as cv
import numpy as np

img = cv.imread('opencv_basics/resources/Photos/park.jpg')
cv.imshow('img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)



# canny - cleaner version of edge detection
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)



# laplacian
lap = cv.Laplacian(gray,ddepth=cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)



# Sobel 
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)



cv.waitKey(0)