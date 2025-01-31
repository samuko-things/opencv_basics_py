import cv2
import numpy as np
from utils import stackImages


path = "opencv_basics/resources/lena.png"
img = cv2.imread(path)
cv2.imshow("Actual Image", img)

## blurring is used to smoothout noise in images
## kernel size is the filter size - the more the kernel size the more the blur

## averaging blur
avg_blur = cv2.blur(img, ksize=(5,5))
cv2.imshow("Average Blur", avg_blur)

## gaussian blur - has more natural blurring effect
gaus_blur = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0, sigmaY=0)
cv2.imshow("Gaussian Blur", gaus_blur)

## median blur - it tends to be more effective than avg and gaus in reduction of substantial amt of noise
## used in more advanced cv projects. enter kernel size as a single number
median_blur = cv2.medianBlur(img, ksize=5)
cv2.imshow("Median Blur", median_blur)


## bi-lateral blur - its the most effective
## it blurs and retains the edges in the image
bi_blur = cv2.bilateralFilter(img, d=15, sigmaColor=45, sigmaSpace=45) # d is diameter
cv2.imshow("Bi-lateral Blur", bi_blur)

# imgStack = stackImages(0.5,([[img,gray,blur],[canny,blank,blank2]]))
# cv2.imshow("imgStack", imgStack)

while True:
    # press esc key to exit loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break