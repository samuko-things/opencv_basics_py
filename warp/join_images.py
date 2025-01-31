# putiing difernt images in one window ####

# import cv2
# import numpy as np
# from utils import stackImages




# img = cv2.imread("opencv_basics/resources/lena.png")
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# # imgHor = np.hstack((img,img)) # horizontal stack
# # imgVar = np.vstack((img,img)) # vertical stack

# # imgStack = stackImages(0.3, ([img,img,imgGray]))
# imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))



# # cv2.imshow("Horizontal",imgHor)
# # cv2.imshow("Vertical",imgVar)

# cv2.imshow("Stacked_img", imgStack)


# cv2.waitKey(0)



# import cv2

# def empty(a):
#     pass

# cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Trackbars",640,300)
# cv2.createTrackbar("Hue Min","Trackbars",0,179,empty)
# cv2.createTrackbar("Hue Max","Trackbars",179,179,empty)
# cv2.createTrackbar("Sat Min","Trackbars",0,255,empty)
# cv2.createTrackbar("Sat Max","Trackbars",255,255,empty)
# cv2.createTrackbar("Val Min","Trackbars",0,255,empty)
# cv2.createTrackbar("Val Max","Trackbars",255,255,empty)

# ch = None
# while ch != 27:
#     ch = cv2.waitKey(0)



# Demo Trackbar
# importing cv2 and numpy
import cv2
import numpy
 
def nothing(x):
    pass
 
# Creating a window with black image
img = numpy.zeros((300, 512, 3), numpy.uint8)
cv2.namedWindow('image')
 
# creating trackbars for red color change
cv2.createTrackbar('R', 'image', 0, 255, nothing)
 
# creating trackbars for Green color change
cv2.createTrackbar('G', 'image', 0, 255, nothing)
 
# creating trackbars for Blue color change
cv2.createTrackbar('B', 'image', 0, 255, nothing)
 
while(True):
    # show image
    cv2.imshow('image', img)
 
    # for button pressing and changing
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
 
    # get current positions of all Three trackbars
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
 
    # display color mixture
    img[:] = [b, g, r]
    
    print(b,g, r)
 
# close the window
cv2.destroyAllWindows()