#pylint:disable=no-member

import numpy as np
import cv2 as cv


path = "opencv_basics/resources/Faces/val/elton_john/4.jpg"
cascade_path = "opencv_basics/face_recognition/haar_face.xml"


haar_cascade = cv.CascadeClassifier(cascade_path)

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# features = np.load('opencv_basics/face_recognition/features.npy', allow_pickle=True)
# labels = np.load('opencv_basics/face_recognition/labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("opencv_basics/face_recognition/face_trained.yml")

img = cv.imread(path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)




# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
