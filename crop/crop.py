import cv2
import numpy as np

# img = cv2.imread("/home/samuko95/py_dev/opencv_basics/resources/shapes.png")
# print(img.shape) # (height, width, no_of_channel)

# # resize image
# frameWidth = 320
# frameHeight = 240
# imgResize = cv2.resize(img,(frameWidth,frameHeight))
# print(imgResize.shape)


# # crop image
# # take a part of the numpy array
# # [height index_range, width index_range]
# imgCropped = img[10:180,10:130]
# print(imgCropped.shape)


# cv2.imshow("Image",img)
# cv2.imshow("Image Resize",imgResize)
# cv2.imshow("Image Cropped",imgCropped)

# cv2.waitKey(0)

s = [1,0,0,1]
 
# using list comprehension
listToStr = ''.join([str(elem) for elem in s])
 
print(listToStr)