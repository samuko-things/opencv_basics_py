from tkinter import X
import cv2 # OpenCV library
import time

import numpy as np


def splitRawData(data):
    X, y = data[:, 0:-1], data[:, -1]

    if len(X.shape)==1:
        X = np.c_[X] # convert to a column vector
    
    y = np.c_[y] #convert to a column vector

    return X,y


w = 80
h = 60 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


x = np.zeros(((h*w)+1,1))

while cv2.waitKey(1) & 0xFF != ord('q'):
    is_successful, vid = cap.read()
    vid = cv2.resize(vid,(w,h))
    v= cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    v1 = v.reshape(h*w,1).ravel().astype(int)
    v1 = np.append(v1, 1)
    print(x.shape)
    print(v1)
    x = np.concatenate([x, np.c_[v1]], axis=1)
    
    # img = cv2.imread('green_line1.png')
    # vid = cv2.resize(img, (300, 200))
    # cv2.imshow("video", vid)a

    cv2.imshow("video2", v)
    # cv2.imwrite("capture/cam_img{}.png".format(time.time()), vid)


# data = x.T
# X,y = splitRawData(data)
# print(data.shape)
# cv2.imwrite("data.png", X[38,:].reshape(h,w))
# np.savetxt("data1.txt", data.astype(int))

data = np.loadtxt('data1.txt').astype(int)
X,y = splitRawData(data)
print(data.shape)
cv2.imwrite("data.png", X[38,:].reshape(h,w))

cap.release()
cv2.destroyAllWindows()