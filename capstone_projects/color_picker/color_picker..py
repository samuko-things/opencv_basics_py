import cv2
import numpy as np
from utils import stackImages



def nothing(x):
    pass

cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 600, 200)
 
cv2.createTrackbar('hmin', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('hmax', 'Trackbars', 179, 179, nothing)

cv2.createTrackbar('smin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('smax', 'Trackbars', 255, 255, nothing)

cv2.createTrackbar('vmin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('vmax', 'Trackbars', 255, 255, nothing)




frameWidth = 320
frameHeight = 240
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)



while True:
    success, img = cap.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## gaussian blur - has more natural blurring effect
    gaus_blur = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0, sigmaY=0)
    cv2.imshow("Gaussian Blur", gaus_blur)
    
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
    
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # # imgStack = stackImages(0.5,([[img, imgHSV],[mask, result]]))
    # imgStack = stackImages(0.5,([img, imgHSV,mask, result]))
    
    # cv2.imshow("Trackbars", imgStack)
    
    # # for button pressing and changing
    # # press esc to close
    # k = cv2.waitKey(1) & 0xFF
    # if k == 27:
    #     break

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    hStack = np.hstack([img, mask, result])
    
    cv2.imshow('result', hStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()