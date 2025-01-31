# USAGE: You need to specify a filter and "only one" image source
# python ColorPicker.py --filter RGB --image /path/image.png
# or
# python ColorPicker.py --filter HSV --webcam
import time

import cv2
import numpy as np
import argparse
from operator import xor

display_res = (480,320)
# video_file = "line_follower1.mp4"
image_file = "green_img1.png"

def callback(value):
    pass

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True,
                    help='Range filter. RGB or HSV')
    ap.add_argument('-i', '--image', required=False,
                    help='Path to the image')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')
    ap.add_argument('-p', '--preview', required=False,
                    help='Show a preview of the image after applying the mask',
                    action='store_true')
    args = vars(ap.parse_args())

    if not xor(bool(args['image']), bool(args['webcam'])):
        ap.error("Please specify only one image source")

    if not args['filter'].upper() in ['RGB', 'HSV']:
        ap.error("Please speciy a correct filter.")

    return args

def get_trackbar_values(range_filter):
    values = []
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

def main():
    args = get_arguments()
    range_filter = args['filter'].upper()
    print(range_filter)

    if args['image']:
        image = cv2.imread(args['image'])
        image = cv2.resize(image, display_res)

        if range_filter == 'RGB':
            frame_to_thresh = image.copy()
        else:
            frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        camera = cv2.VideoCapture(2)

    setup_trackbars(range_filter)

    while True:
        if args['webcam']:
            ret, image = camera.read()
            image = cv2.resize(image, display_res)

            if not ret:
                break

            if range_filter == 'RGB':
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)
        mask = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        kernel = np.ones((5,5),'int')
        eroded = cv2.erode(mask, kernel, iterations=2)
        dilated = cv2.dilate(eroded,kernel, iterations=4)
        res = cv2.bitwise_and(image,image,mask=dilated)
        ret,thresh = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
        contours,hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        if args['preview']:
            preview = cv2.bitwise_and(image, image, mask=thresh)
            cv2.imshow("Preview", preview)
        else:
            # image = cv2.flip(image, 1)
            cv2.imshow("Original", image)

            # thresh = cv2.flip(thresh, 1)
            cv2.imshow("Thresh", thresh)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break
        
        time.sleep(0.05)
# USAGE: You need to specify a filter and "only one" image source
# python3 ColorPicker.py --filter RGB --image /path/image.png
# or
# python3 ColorPicker.py --filter HSV --webcam
if __name__ == '__main__':
    main()