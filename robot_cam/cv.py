
import cv2
import numpy as np

import time

print_cnt = False

prev_cam_array = "00000"
cam_array_buffer = "00000"

display_resolution = (160,120)
display_width = display_resolution[0]
display_height = display_resolution[1]

cap = cv2.VideoCapture(2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)







def convert_to_cam_array(count, threshold):
    global prev_cam_array

    cam_array = ""

    for count_val in count:
        if count_val >= threshold:
            cam_array +="1"
        else:
            cam_array +="0"
    
    return cam_array



def cam_ir_sense(masked_img, r_h, r_w, h, w, d, rect_point, threshold):
    rect_h = r_h
    rect_w = r_w

    count = [0,0,0,0,0]

    for i in range(5):

        for j in range(h, h+rect_h):
            for k in range(rect_point[i][0], rect_point[i][0]+rect_w):
                if masked_img[j][k] == 255:
                    count[i]+=1
    
    cam_array = convert_to_cam_array(count, threshold)
    return cam_array


def cam_color_detect(masked_img, r_h, r_w, h, w, d, rect_point, threshold):
    color_found = "no_stop"

    rect_h = r_h
    rect_w = r_w

    count = [0,0,0,0,0]

    for i in range(5):

        for j in range(h, h+rect_h):
            for k in range(rect_point[i][0], rect_point[i][0]+rect_w):
                if masked_img[j][k] == 255:
                    count[i]+=1

    cnt = 0 
    for count_val in count:
        if count_val >= threshold:
            cnt +=1
    
    if cnt>=3:
        color_found = "stop"

    return color_found


def angle_check(img, masked_img, display_height, display_width):
    ang = 0
    error = 0
    contours,hier = cv2.findContours(masked_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
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

        cv2.drawContours(img, [box], 0, (0,0,255), 1)
        
        angle_str = "ang= "+str(ang)
        error_str = "err= "+str(error)
        cv2.putText(img, angle_str, (10,int(display_height/8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(img, error_str, (10,int(display_height*7/8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        start_point =  ( int(x+int(w/2)), int(y+int(h/3)) ) 
        end_point = ( int(x+int(w/2)), int(y+int(2*h/3)) )  
        cv2.line(img, start_point, end_point, (0,255,0), 1)

    return img, ang, error



def main():
    while cv2.waitKey(1) & 0xFF != ord('q'):
        start_time = time.time()

        is_successful, img = cap.read()
        img = cv2.resize(img, display_resolution)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #####################################################################################################
        lower_blue = np.array([60,0,240])
        upper_blue = np.array([160,255,255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        blue_kernel = np.ones((5,5),'int')
        blue_eroded = cv2.erode(blue_mask, blue_kernel, iterations=2)
        blue_dilated = cv2.dilate(blue_eroded,blue_kernel, iterations=4)
        blue_res = cv2.bitwise_and(img,img,mask=blue_dilated)
        _,blue_thresh = cv2.threshold(cv2.cvtColor(blue_res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
        # contours,hier = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #######################################################################################################


        ########################################################################################################
        lower_green = np.array([15,15,15])
        upper_green = np.array([102,255,255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green) 

        green_kernel = np.ones((5,5),'int')
        green_eroded = cv2.erode(green_mask, green_kernel, iterations=2)
        green_dilated = cv2.dilate(green_eroded,green_kernel, iterations=4)
        green_res = cv2.bitwise_and(img,img,mask=green_dilated)
        _,green_thresh = cv2.threshold(cv2.cvtColor(green_res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
        green_invert = cv2.bitwise_not(green_thresh)
        # contours,hier = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #########################################################################################################


        #########################################################################################################
        rect_h = 10
        rect_w = 10
        h = 55
        w = 15
        d = 30

        rect_point = []
        for i in range(5):
            rect_point.append( (w+(d*i),h) )
        
        for point in rect_point:
            cv2.rectangle(img, point, (point[0]+rect_w, point[1]+rect_h), (0,0,255), 1)
        ########################################################################################################


        cam_array = cam_ir_sense(blue_thresh, rect_h, rect_w, h, w, d, rect_point, threshold=80)
        stop_detected = cam_color_detect(green_thresh, rect_h, rect_w, h, w, d, rect_point, threshold=98)
        img, angle, error = angle_check(img, blue_thresh, display_height, display_width)
         
        cv2.imshow("img", img)

        str_val = ""
        str_val+=cam_array
        str_val+=","
        str_val+=stop_detected
        str_val+= ","
        str_val+=str(angle)

        print(str_val, round((time.time()-start_time), 3))


    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()









































# import cv2
# import numpy as np

# import time

# print_cnt = False

# display_resolution = (160,120)
# display_width = display_resolution[0]
# display_height = display_resolution[1]

# cap = cv2.VideoCapture(2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)


# def convert_to_cam_array(count1, count2, count3):
#     threshold = 20
#     cam_array = ""

#     if count1 > threshold:
#         cam_array +="1"
#     else:
#         cam_array +="0"

#     if count2 > threshold:
#         cam_array +="1"
#     else:
#         cam_array +="0"

#     if count3 > threshold:
#         cam_array +="1"
#     else:
#         cam_array +="0"

#     return cam_array


# while cv2.waitKey(1) & 0xFF != ord('q'):
#     start_time = time.time()
#     # img = cv2.imread("green_line1.png")

#     is_successful, img = cap.read()
#     # img = cv2.resize(img, display_resolution)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     lower_green = np.array([60,0,190])
#     upper_green = np.array([160,255,255])
#     mask = cv2.inRange(hsv, lower_green, upper_green) 

#     kernel = np.ones((5,5),'int')
#     eroded = cv2.erode(mask, kernel, iterations=2)
#     dilated = cv2.dilate(eroded,kernel, iterations=4)
#     res = cv2.bitwise_and(img,img,mask=dilated)
#     ret,threshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
#     contours,hier = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#     rect_h = 20
#     rect_w = 20
#     h = 50
#     w = 20
#     d = 50
#     rect1_point = (w,h)
#     rect2_point = (w+d,h)
#     rect3_point = (w+(d*2),h)
#     cv2.rectangle(img, rect1_point, (rect1_point[0]+rect_w, rect1_point[1]+rect_h), (255,0,0), 4)
#     cv2.rectangle(img, rect2_point, (rect2_point[0]+rect_w, rect2_point[1]+rect_h), (255,0,0), 4)
#     cv2.rectangle(img, rect3_point, (rect3_point[0]+rect_w, rect3_point[1]+rect_h), (255,0,0), 4)

#     count1 = 0
#     for i in range(h, h+rect_h+1):
#         for j in range(rect1_point[0], rect1_point[0]+rect_w+1):
#             if threshed[i][j] == 255:
#                 count1+=1

#     count2 = 0
#     for i in range(h, h+rect_h+1):
#         for j in range(rect2_point[0], rect2_point[0]+rect_w+1):
#             if threshed[i][j] == 255:
#                 count2+=1

#     count3 = 0
#     for i in range(h, h+rect_h+1):
#         for j in range(rect3_point[0], rect3_point[0]+rect_w+1):
#             if threshed[i][j] == 255:
#                 count3+=1

#     # count4 = 0
#     # for i in range(h, h+rect_h+1):
#     #     for j in range(rect2_point[0], rect2_point[0]+rect_w+1):
#     #         if threshed[i][j] == 255:
#     #             count4+=1

#     # count5 = 0
#     # for i in range(h, h+rect_h+1):
#     #     for j in range(rect3_point[0], rect3_point[0]+rect_w+1):
#     #         if threshed[i][j] == 255:
#     #             count5+=1
    
#     cam_array = convert_to_cam_array(count1, count2, count3)
 	
#     cv2.imshow("img", img)
#     print(cam_array, time.time()-start_time)

#     # if not print_cnt:
#     #     print(mask[0])
#     #     print_cnt = True
 
# # cv2.imwrite("my_img.png", vid)

# # cap.release()
# cv2.destroyAllWindows()



# rect1_point = (w,h)
        # rect2_point = (w+d,h)
        # rect3_point = (w+(d*2),h)
        # rect4_point = (w+(d*3),h)
        # rect5_point = (w+(d*4),h)


# cv2.rectangle(img, rect1_point, (rect1_point[0]+rect_w, rect1_point[1]+rect_h), (255,0,0), 2)
        # cv2.rectangle(img, rect2_point, (rect2_point[0]+rect_w, rect2_point[1]+rect_h), (255,0,0), 2)
        # cv2.rectangle(img, rect3_point, (rect3_point[0]+rect_w, rect3_point[1]+rect_h), (255,0,0), 2)
        # cv2.rectangle(img, rect4_point, (rect4_point[0]+rect_w, rect4_point[1]+rect_h), (255,0,0), 2)
        # cv2.rectangle(img, rect5_point, (rect5_point[0]+rect_w, rect5_point[1]+rect_h), (255,0,0), 2)



    # count1 = 0
    # for i in range(h, h+rect_h):
    #     for j in range(rect1_point[0], rect1_point[0]+rect_w):
    #         if masked_img[i][j] == 255:
    #             count1+=1

    # count2 = 0
    # for i in range(h, h+rect_h):
    #     for j in range(rect2_point[0], rect2_point[0]+rect_w):
    #         if masked_img[i][j] == 255:
    #             count2+=1

    # count3 = 0
    # for i in range(h, h+rect_h):
    #     for j in range(rect3_point[0], rect3_point[0]+rect_w):
    #         if masked_img[i][j] == 255:
    #             count3+=1

    # count4 = 0
    # for i in range(h, h+rect_h):
    #     for j in range(rect4_point[0], rect4_point[0]+rect_w):
    #         if masked_img[i][j] == 255:
    #             count4+=1

    # count5 = 0
    # for i in range(h, h+rect_h):
    #     for j in range(rect5_point[0], rect5_point[0]+rect_w):
    #         if masked_img[i][j] == 255:
    #             count5+=1


# if count2 >= threshold:
#         cam_array +="1"
#     else:
#         cam_array +="0"

#     if count3 >= threshold:
#         cam_array +="1"
#     else:
#         cam_array +="0"

#     if count4 > threshold:
#         cam_array +="1"
#     else:
#         cam_array +="0"

#     if count5 > threshold:
#         cam_array +="1"
#     else:
#         cam_array +="0"

#     new_cam_array = ""
#     if (cam_array == "00000") or (cam_array == "11111") or (cam_array == "00001") or (cam_array == "00011") or (cam_array == "00010") or (cam_array == "00110") or (cam_array == "00100") or (cam_array == "01100") or (cam_array == "01000") or (cam_array == "11000") or (cam_array == "10000") :
#         new_cam_array = cam_array
#         prev_cam_array = cam_array
#     else:
#         new_cam_array = prev_cam_array

#     return new_cam_array