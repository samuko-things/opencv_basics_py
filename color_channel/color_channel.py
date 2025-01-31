import cv2
import numpy as np


path = "opencv_basics/resources/test.png"
img = cv2.imread(path) # image is read as bgr image
blank = np.zeros(img.shape[:2], dtype='uint8')


b,g,r = cv2.split(img) # resulting images are grayscales

blue = cv2.merge([b,blank,blank])
green = cv2.merge([blank,g,blank])
red = cv2.merge([blank,blank,r])

merge = cv2.merge([b,g,r])

print(img.shape)
print(b.shape, g.shape, r.shape)

cv2.imshow("b", blue)
cv2.imshow("g", green)
cv2.imshow("r", red)

cv2.imshow("merge", merge)


while True:
    # press esc key to exit loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


## darker region in the resulting grayscale image indicates that the color is less there
## lighter regions indicates that the color is more there