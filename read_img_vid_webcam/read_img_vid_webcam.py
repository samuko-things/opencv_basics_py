# #### READ IMAGES WITH OPENCV ############

# import cv2

# # LOAD AN IMAGE USING 'IMREAD'
# img = cv2.imread("opencv_basics/resources/lena.png")
# # DISPLAY
# cv2.imshow("Lena Soderberg",img)
# cv2.waitKey(0) ## wait forever until you close the image window




# ##### READ VIDEO WITH OPENCV ############
# import cv2

# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture("opencv_basics/resources/robotic_hand.mp4")
# while True:
#     is_successful, img = cap.read()
#     img = cv2.resize(img, (frameWidth, frameHeight))
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'): ## exit loop if q is pressed - quit
#         break





#### READ WEB CAM ###################

import cv2
import time

frameWidth = 640
frameHeight = 480
brightness = 100
cap = cv2.VideoCapture(2)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

current_time = time.time()
prev_time = time.time()

while True:
    current_time = time.time()
    is_successful, img = cap.read()
    cv2.imshow("webcam", img)
    
    fps = 1/(current_time-prev_time)
    prev_time=current_time
    print("fps = ", int(fps))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

