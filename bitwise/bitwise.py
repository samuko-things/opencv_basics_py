import cv2
import numpy as np
from utils import stackImages


path = "opencv_basics/resources/lena.png"
img = cv2.imread(path)
# cv2.imshow("Actual Image", img)


blank = np.zeros(shape=(400,400), dtype='uint8')

rectangle = cv2.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv2.circle(blank.copy(), (200,200), 200, 255, -1)

cv2.imshow("rectangle", rectangle)
cv2.imshow("circle", circle)
# imgStack = stackImages(0.5,([[img,gray,blur],[canny,blank,blank2]]))
# cv2.imshow("imgStack", imgStack)


# bitwise_and - returns intersection
bit_and = cv2.bitwise_and(rectangle,circle)
cv2.imshow("AND", bit_and)

# bitwise_or - returns both intersecting and non intersectiong region
bit_or = cv2.bitwise_or(rectangle,circle)
cv2.imshow("OR", bit_or)

# bitwise_xor - returns non intersectiong region
bit_xor = cv2.bitwise_xor(rectangle,circle)
cv2.imshow("XOR", bit_xor)

# bitwise_not
bit_not = cv2.bitwise_not(circle)
cv2.imshow("NOT", bit_not)

while True:
    # press esc key to exit loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break