import cv2
import numpy as np


def detect_line_segments(edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image



print_cnt = False



cap = cv2.VideoCapture(2)
# img = cv2.imread("green_img1.png")
# img = cv2.resize(img, (480, 320))

while cv2.waitKey(1) & 0xFF != ord('q'):
    is_successful, img = cap.read()
    # img = cv2.resize(img, display_resolution)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([60,0,240])
    upper_blue = np.array([160,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 

    # lower_green = np.array([40,0,0])
    # upper_green = np.array([100,255,255])
    # mask = cv2.inRange(hsv, lower_green, upper_green)
    # # greenline= cv2.inRange(img, (0,255,0), (50,255,50))

    # cv2.imshow("img", mask)

    kernel = np.ones((5,5),'int')
    dilated = cv2.erode(mask, kernel, iterations=3)
    dilated = cv2.dilate(dilated,kernel, iterations=5)
    res = cv2.bitwise_and(img,img,mask=dilated)
    ret,threshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
    contours,hier = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:	 
        blackbox = cv2.minAreaRect(contours[0])
        (x_min, y_min), (w_min, h_min), ang = blackbox
        if ang < -45 :
            ang = 90 + ang
        if w_min < h_min and ang > 0:	  
            ang = (90-ang)*-1
        if w_min > h_min and ang < 0:
            ang = 90 + ang	  
        setpoint = 320
        error = int(x_min - setpoint) 
        ang = int(ang)	 
        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),3)	 
        cv2.putText(img,str(ang),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img,str(error),(10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(img, (int(x_min),200 ), (int(x_min),250 ), (255,0,0),3)
	 	

    # if len(contours) > 0 :
    #     x,y,w,h = cv2.boundingRect(contours[0])
        
    #     start_point =  ( (x+int(w/2)), (y+int(h/3)) ) 
    #     end_point = ( (x+int(w/2)), (y+int(2*h/3)) )
        
    #     cv2.line(img, start_point, end_point, (0,0,255), 2)

    #     if print_cnt == False:
    #        print(x,y,w,h)
    #        print_cnt = True
        
            
    # edges = cv2.Canny(mask, 200, 400)

    # lines = detect_line_segments(edges)

    # lines_image = display_lines(img, lines)

    # cv2.imshow("img", img)
    cv2.imshow("img", img)
 
# cv2.imwrite("my_img.png", vid)

# cap.release()
cv2.destroyAllWindows()













# cv2.drawContours(img, contours, 0, (0, 255, 0), 1)

#     # if print_cnt == False:
#     #     print(list(contours[0][0]))
#     #     print(len(contours[0]))
#     #     # x,y,w,h = cv2.boundingRect(contours[0])
#     #     # print(x,y,w,h)
#     #     print_cnt = True

#     if len(contours) > 0 :
#         print(len(contours[0]))
#         x,y,w,h = cv2.boundingRect(contours[0])
#         # x,y,w,h=230,0,164,320
#         print(x,y,w,h)

#         start_point =  ( (x+int(w/2)), (y+int(h/3)) ) 
#         end_point = ( (x+int(w/2)), (y+int(2*h/3)) )
#         # start_point =  (x, y) 
#         # end_point = (x+w, y+h)
#         cv2.line(img, start_point, end_point, (0,0,255), 2)

#         # if print_cnt == False:
#         #    print(x,y,w,h)
#         #    print_cnt = True













# from picamera.array import PiRGBArray
# from picamera import PiCamera
# import time
# import cv2
# import numpy as np
# import RPi.GPIO as GPIO

# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(40, GPIO.OUT)
# GPIO.output(40, GPIO.HIGH)

# camera = PiCamera()
# camera.resolution = (640, 360)
# camera.rotation = 180
# rawCapture = PiRGBArray(camera, size=(640, 360))
# time.sleep(0.1)

# for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):	
# 	image = frame.array
# 	roi = image[200:250, 0:639]
# 	Blackline= cv2.inRange(roi, (0,0,0), (50,50,50))
# 	kernel = np.ones((3,3), np.uint8)
# 	Blackline = cv2.erode(Blackline, kernel, iterations=5)
# 	Blackline = cv2.dilate(Blackline, kernel, iterations=9)	
# 	img,contours, hierarchy = cv2.findContours(Blackline.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
# 	if len(contours) > 0 :
# 	   x,y,w,h = cv2.boundingRect(contours[0])	   
# 	   cv2.line(image, (x+(w/2), 200), (x+(w/2), 250),(255,0,0),3)
# 	cv2.imshow("orginal with line", image)	
# 	rawCapture.truncate(0)	
# 	key = cv2.waitKey(1) & 0xFF	
# 	if key == ord("q"):
# 		break

# GPIO.output(40, GPIO.LOW)