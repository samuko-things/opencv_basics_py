import cv2
import numpy as np
from utils import stackImages


path = "opencv_basics/resources/lena.png"
img = cv2.imread(path)
blank, blank2 = np.zeros_like(img), np.zeros_like(img)

#pre-procesing

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conver to grayscale
blur = cv2.GaussianBlur(gray,ksize=(9,9),sigmaX=3) # add blur to grayscale img
canny = cv2.Canny(image=blur,threshold1=50,threshold2=50) # edge detect with the blur image

#### cv2.RETR_LIST, cv2.RETR_TREE   |  cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# area = cv2.contourArea(contours)
print(f"{len(contours)} contours found")

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area>50: # for filtering noise
        print(f"{area} area found")
        cv2.drawContours(blank, cnt, contourIdx=-1, color=(255,0,0),thickness=2)
        
        #calc curved length to approx the corners of our shapes
        peri = cv2.arcLength(cnt,True)
        # print(peri)
        #approx howmany corner point we have
        corner_pnts = cv2.approxPolyDP(cnt,0.02*peri,True)
        # print(len(corner_pnts))
        # create bounding box around detected objects
        x,y,w,h = cv2.boundingRect(corner_pnts)
        # draw the bounding rectangle
        cv2.rectangle(blank,(x,y),(x+w,y+h),(0,255,0),2)

# cv2.drawContours(blank, contours=contours, contourIdx=-1, color=(0,255,0), thickness=2)

imgStack = stackImages(0.5,([[img,gray,blur],[canny,blank,blank2]]))
cv2.imshow("imgStack", imgStack)

while True:
    # press esc key to exit loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break