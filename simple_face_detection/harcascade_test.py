import cv2 as cv
import numpy as np
from utils import stackImages

####### simple face detection with cascade methods #######
# only for detction, not recognition

path = "opencv_basics/resources/Photos/group 2.jpg"
cascade_path = "opencv_basics/face_detection/haar_face.xml"
# cascade_path = "opencv_basics/resources/haarcascade_frontalface_default.xml"



img = cv.imread(path)
cv.imshow('Group of 5 people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)


haar_cascade = cv.CascadeClassifier(cascade_path)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)



while True:
    # press esc key to exit loop
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
