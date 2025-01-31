import cv2
import numpy as np
import matplotlib.pyplot as plt


path = "opencv_basics/resources/lena.png"
img = cv2.imread(path) # image is read as bgr image
blank, blank2 = np.zeros_like(img), np.zeros_like(img)

#pre-procesing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conver to grayscale

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to hsv format

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # convert to lab format

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to rgb format

cv2.imshow("imgStack", rgb)

plt.imshow(rgb)
plt.show()

while True:
    # press esc key to exit loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


## POSIBLE CONVERTION ##
## hsv to bgr and vice versa
## lab to bgr and vice versa
## gray to bgr and vice versa
## gray to bgr to hsv and vice versa