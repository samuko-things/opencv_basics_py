import cv2
import numpy as np
from utils import stackImages

####### simple face detection with cascade methods #######
#although not the most accurate one but its very fast

frameWidth = 640
frameHeight = 480
brightness = 100
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)


cascade_path = "opencv_basics/resources/haarcascade_frontalface_default.xml"
faceCascade= cv2.CascadeClassifier(cascade_path)




while True:
    is_successful, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(imgGray,1.1,4) # imgGray, scalefactor, minimum labels
 
    # create bounding box around the faces you have detected
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow("webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


