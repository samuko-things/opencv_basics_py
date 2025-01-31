import cv2
import numpy as np

img = cv2.imread("opencv_basics/resources/lena.png")

# conver to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur image with guasian blur
ksize_blur = (9,9) # kernel size must be odd number
imgBlur = cv2.GaussianBlur(imgGray,ksize=ksize_blur,sigmaX=0)

# to find edges in an image we use the canny edge detector
# adjust the threshold to your taste
# usally when using the canny, we use blur image input
imgCanny = cv2.Canny(image=img,threshold1=150,threshold2=300)

# add diliation to increase the thickness of the detected edge
# more iteration means more thickness
# the lesser the kernel size, the smaller the intial thickness
kernel_dilate = np.ones(shape=(3,3), dtype=np.uint8) # odd no size
imgDilate = cv2.dilate(imgCanny, kernel=kernel_dilate, iterations=1)


# remove thickness with erode (opposite of dilation)
# more iteration means more erosion
# the lesser the kernel size, the smaller the intial erosion
kernel_erode = np.ones(shape=(3,3), dtype=np.uint8) # odd no size
imgErode = cv2.erode(imgDilate, kernel=kernel_erode, iterations=2)


cv2.imshow("Gray Image",imgGray)
cv2.imshow("Blur Image",imgBlur)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Dilated Image",imgDilate)
cv2.imshow("Eroded Image",imgErode)
cv2.waitKey(0)