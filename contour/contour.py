import cv2
import numpy as np
from utils import stackImages



def get_contours(originalImg, img):
    imgContour = originalImg.copy()
    # mat = image
    # mode = cv2.RETR_EXTERNAL # to retrieve the extreme outer contours
    # method = cv2.CHAIN_APPROX_NONE # request for all the coutours fond without approximation
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        # cv2.drawContours(imgContour, cnt, contourIdx=-1, color=(255,0,0),thickness=5)
        if area>500: # for filtering noise
            cv2.drawContours(imgContour, cnt, contourIdx=-1, color=(255,0,0),thickness=5)
            #calc curved length to approx the corners of our shapes
            peri = cv2.arcLength(cnt,True)
            # print(peri)
            #approx howmany corner point we have
            corner_pnts = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(corner_pnts))
            # cv2.drawContours(imgContour, corner_pnts, contourIdx=-1, color=(255,0,0),thickness=5)
            # create bounding box around detected objects
            x,y,w,h = cv2.boundingRect(corner_pnts)
            # draw the bounding rectangle
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            
            
            objCor = len(corner_pnts)
            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"
            
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)
    
    return imgContour
            
            
            
            
            



path = "opencv_basics/resources/shapes.png"
img = cv2.imread(path)

#pre-procesing

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conver to grayscale
imgBlur = cv2.GaussianBlur(imgGray,ksize=(7,7),sigmaX=1) # add blur to grayscale img
imgCanny = cv2.Canny(image=imgBlur,threshold1=50,threshold2=50) # edge detect with the blur image
imgContour = get_contours(img, imgCanny)


imgBlank = np.zeros_like(img)
imgStack = stackImages(0.5,([[img,imgGray,imgBlur],[imgCanny,imgContour,imgBlank]]))
cv2.imshow("imgStack", imgStack)

while True:
    # press esc key to exit loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
