import cv2
import numpy as np
from utils import stackImages



def nothing(x):
    pass

cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Trackbars", 600, 200)
 
cv2.createTrackbar('hmin', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('hmax', 'Trackbars', 179, 179, nothing)

cv2.createTrackbar('smin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('smax', 'Trackbars', 255, 255, nothing)

cv2.createTrackbar('vmin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('vmax', 'Trackbars', 255, 255, nothing)


img = cv2.imread("opencv_basics/resources/lambo.png")
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    hmin = cv2.getTrackbarPos('hmin', 'Trackbars')
    hmax = cv2.getTrackbarPos('hmax', 'Trackbars')
    
    smin = cv2.getTrackbarPos('smin', 'Trackbars')
    smax = cv2.getTrackbarPos('smax', 'Trackbars')
    
    vmin = cv2.getTrackbarPos('vmin', 'Trackbars')
    vmax = cv2.getTrackbarPos('vmax', 'Trackbars')
    
    # print(hmin,hmax,smin,smax,vmin,vmax)
    
    lower_limit = np.array([hmin, smin, vmin])
    upper_limit = np.array([hmax, smax, vmax])
    
    mask = cv2.inRange(imgHSV, lower_limit, upper_limit)
    
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    
    # imgStack = stackImages(0.5,([[img, imgHSV],[mask, imgResult]]))
    imgStack = stackImages(0.5,([img, imgHSV,mask, imgResult]))
    
    cv2.imshow("Trackbars", imgStack)
    
    # for button pressing and changing
    # press esc to close
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break