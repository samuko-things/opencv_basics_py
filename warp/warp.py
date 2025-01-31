import cv2
import numpy as np


img = cv2.imread("opencv_basics/resources/cards.jpg")




width,height = 250,350
# width,height = 500,700

# define the four corner point of the card in the image
# u can use apps like paint to get these points
card_pts = np.float32([[111,219],[287,188],[154,482],[352,440]])

# define the points you want to transform to - a rectangular image
new_card_pts = np.float32([[0,0],[width,0],[0,height],[width,height]])

# create the transformation matrix
transformationMatrix = cv2.getPerspectiveTransform(card_pts, new_card_pts)

# form the warpped image with the transformation matrix
# img, transformationMatrix, the width and height of the warped img frame
imgWarp = cv2.warpPerspective(img,transformationMatrix,(width,height))





cv2.imshow("Image", img)
cv2.imshow("Warpped Image", imgWarp)

cv2.waitKey(0)