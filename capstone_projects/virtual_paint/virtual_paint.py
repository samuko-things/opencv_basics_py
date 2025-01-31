import cv2
import numpy as np
from utils import stackImages





frameWidth = 640
frameHeight = 480
brightness = 100
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)



myColors = [
    [94,58,12,125,255,255], # blue marker
    [0,38,34,25,255,255], # red marker
]

color_values = [
    (255,0,0),  # blue marker
    (0,0,255)   # red marker
]

myPoints =  []  ## [x , y , colorId ]




def findColor(img, hsv_color, color_val):
    ## gaussian blur - has more natural blurring effect
    gaus_blur = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=2, sigmaY=0)
    # cv2.imshow("Gaussian Blur", gaus_blur)
    imgHSV = cv2.cvtColor(gaus_blur, cv2.COLOR_BGR2HSV)
    
    newPoints = []
    
    for colors in hsv_color:
        lower_limit = np.array(colors[0:3])
        upper_limit = np.array(colors[3:6])
        
        mask = cv2.inRange(imgHSV, lower_limit, upper_limit)
        
        kernel_dilate = np.ones(shape=(5,5), dtype=np.uint8) # odd no size
        imgDilate = cv2.dilate(mask, kernel=kernel_dilate, iterations=1)
        
        x,y = get_contours(img, imgDilate)
        
        newPoints.append([x,y,hsv_color.index(colors)])
        
    return newPoints



def get_contours(originalImg, img):   
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # cv2.drawContours(imgContour, cnt, contourIdx=-1, color=(255,0,0),thickness=5)
        if area>1000: # for filtering noise
            # cv2.drawContours(imgContour, cnt, contourIdx=-1, color=(255,0,0),thickness=5)
            #calc curved length to approx the corners of our shapes
            peri = cv2.arcLength(cnt,True)
            #approx howmany corner point we have
            corner_pnts = cv2.approxPolyDP(cnt,0.02*peri,True)
            # create bounding box around detected objects
            x,y,w,h = cv2.boundingRect(corner_pnts)
            # draw the bounding rectangle
            # cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            
    return x+w//2, y





def drawOnCanvas(pnts, colorVal):
    for pnt in pnts:
        cv2.circle(imgContour, (pnt[0],pnt[1]), 10, colorVal[pnt[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    # cv2.imshow("result", img)
    imgContour = img.copy()
    # imgContour = np.zeros(shape=(512,512), dtype=np.uint8)
    imgContour.fill(255)
    
    
    newPoints = findColor(img,myColors, color_values)
    
    if len(newPoints)!=0:
        for newPnt in newPoints:
            myPoints.append(newPnt)
            
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, color_values)
    
    cv2.imshow("Result", imgContour)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()