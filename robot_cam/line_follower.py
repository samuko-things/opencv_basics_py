import cv2
import numpy as np

import time

print_cnt = False

display_resolution = (160,120)
display_width = display_resolution[0]
display_height = display_resolution[1]

# cap = cv2.VideoCapture("line_follower11.mp4")
# img = cv2.imread("green_img1.png")
# img = cv2.resize(img, (display_width, display_height))

while cv2.waitKey(1) & 0xFF != ord('q'):
    start_time = time.time()
    img = cv2.imread("green_line1.png")

    # is_successful, img = cap.read()
    img = cv2.resize(img, display_resolution)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40,36,47])
    upper_green = np.array([160,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green) 

    # is_successful, img = cap.read()
    # img = cv2.resize(img, display_resolution)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_green = np.array([45,48,46])
    # upper_green = np.array([80,190,220])
    # mask = cv2.inRange(hsv, lower_green, upper_green)  

    kernel = np.ones((5,5),'int')
    eroded = cv2.erode(mask, kernel, iterations=2)
    dilated = cv2.dilate(eroded,kernel, iterations=4)
    res = cv2.bitwise_and(img,img,mask=dilated)
    ret,threshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
    contours,hier = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:	 
        x,y,w,h = cv2.boundingRect(contours[0])

        min_rect = cv2.minAreaRect(contours[0])
        (x_min, y_min), (w_min, h_min), ang = min_rect
        
        if ang < -45:
            ang = 90+ang
        if w_min < h_min and ang > 0:
            ang = (90-ang)*-1
        if w_min > h_min and ang < 0:
            ang = 90+ang
        ang = int(ang)

        setpoint = int(display_width/2)
        error = int(x_min - setpoint)

        box = cv2.boxPoints(min_rect)
        box = np.int0(box)

        cv2.drawContours(img, [box], 0, (0,0,255), 2)
        
        angle_str = "ang= "+str(ang)
        error_str = "err= "+str(error)
        cv2.putText(img, angle_str, (10,int(display_height/8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(img, error_str, (10,int(display_height*7/8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        start_point =  ( int(x+int(w/2)), int(y+int(h/3)) ) 
        end_point = ( int(x+int(w/2)), int(y+int(2*h/3)) )  
        cv2.line(img, start_point, end_point, (255,0,0), 2)
	 	
    cv2.imshow("img", img)

    print("cool", time.time()-start_time)
 
# cv2.imwrite("my_img.png", vid)

# cap.release()
cv2.destroyAllWindows()

