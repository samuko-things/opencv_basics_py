import cv2
import numpy as np
from utils import stackImages




img_path = "opencv_basics/resources/paper.jpg"

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5,5), 1)
    canny = cv2.Canny(blur,200,200)
    kernel = np.ones((5,5))
    dilate = cv2.dilate(canny,kernel, iterations=2)
    erode = cv2.erode(dilate, kernel, iterations=1)
    
    return erode



def get_contours(img):   
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    maxArea = 0
    biggest = np.array([]) # store the biggest shape's corner_pnts in the image
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # cv2.drawContours(imgContour, cnt, contourIdx=-1, color=(255,0,0),thickness=5)
        if area>5000: # for filtering noise
            # cv2.drawContours(imgContour, cnt, contourIdx=-1, color=(255,0,0),thickness=5)
            #calc curved length to approx the corners of our shapes
            peri = cv2.arcLength(cnt,True)
            #approx howmany corner point we have
            corner_pnts = cv2.approxPolyDP(cnt,0.02*peri,True)
            # cv2.drawContours(imgContour, corner_pnts, contourIdx=-1, color=(255,0,0),thickness=5)
            
            if area>maxArea and len(corner_pnts)==4:
                biggest = corner_pnts
                maxArea = area
            # print(len(corner_pnts))
            
            # # create bounding box around detected objects
            # x,y,w,h = cv2.boundingRect(corner_pnts)
            # # draw the bounding rectangle
            # cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            
    cv2.drawContours(imgContour, biggest, contourIdx=-1, color=(255,0,0),thickness=15)     
    return biggest
     
     
     


def getWarp(img, pnts):
    width,height = img.shape[1], img.shape[0]
    
    # define the four corner point of the card in the image
    # u can use apps like paint to get these points
    card_pts = np.float32(pnts)

    # define the points you want to transform to - a rectangular image
    new_card_pts = np.float32([[0,0],[width,0],[0,height],[width,height]])

    # create the transformation matrix
    transformationMatrix = cv2.getPerspectiveTransform(card_pts, new_card_pts)

    # form the warpped image with the transformation matrix
    # img, transformationMatrix, the width and height of the warped img frame
    imgWarp = cv2.warpPerspective(img,transformationMatrix,(width,height))  
    
    
    # crop the image a little
    imgCrop = imgWarp[20:imgWarp.shape[0]-20, 20:imgWarp.shape[1]-20]   # [height, width]
    imgCrop = cv2.resize(imgCrop, (img.shape[1], img.shape[0])) # (width, height)
    
    return imgCrop  




def reorder(pnts):
    pnts = pnts.reshape(4,2)
    newPnts = pnts.copy()
    add = np.sum(pnts,axis=1) # (sum horizontally - i.e by each row)
    # print(add)
    
    newPnts[0] = pnts[np.argmin(add)] # np.argmin gets index of the minimum val in the add array
    newPnts[3] = pnts[np.argmax(add)] # np.argmin gets index of the maximum val in the add array
    
    diff = np.diff(pnts,axis=1) # (subtract horizontally - i.e by each row)
    # print(diff)
    
    newPnts[1] = pnts[np.argmin(diff)]
    newPnts[2] = pnts[np.argmax(diff)]
    
    return newPnts
    
     
     
     




       

while True:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (480,640))
    # print(img.shape)
    imgContour = img.copy()
    
    thresh = preprocess(img)
    biggest = get_contours(thresh)
    # reorder(biggest)
    
    if biggest.size != 0:
        imgWarp = getWarp(img, reorder(biggest))
        
        imgStack = stackImages(0.6,([[img, thresh],
                                    [imgContour, imgWarp]]))
        
    else:
        imgStack = stackImages(0.6,([[img, thresh],
                                    [img, img]]))
    cv2.imshow("imgStack", imgStack)
        
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break