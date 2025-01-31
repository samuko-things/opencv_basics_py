import cv2
import numpy as np
from utils import stackImages


path = "opencv_basics/resources/lena.png"
img = cv2.imread(path)
# cv2.imshow("Actual Image", img)


blank = np.zeros(img.shape[:2], dtype='uint8')
cv2.imshow('Blank Image', blank)

circle = cv2.circle(blank.copy(), (img.shape[1]//2 + 45,img.shape[0]//2), 100, 255, -1)

rectangle = cv2.rectangle(blank.copy(), (30,30), (370,370), 255, -1)

weird_shape = cv2.bitwise_and(circle,rectangle)
cv2.imshow('Weird Shape', weird_shape)


# mask is a 2d-1channel black and white image with the same dimension as the input img
masked = cv2.bitwise_and(img,img,mask=weird_shape) 
cv2.imshow('Weird Shaped Masked Image', masked)



while True:
    # press esc key to exit loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break