import cv2
import numpy as np

#black image
img = np.zeros(shape=(512,512,3), dtype=np.uint8)
print(img.shape) # (height, width, no_of_channel)


#blue image

# select a section of the 2d frame
# [height index_range(:), width index_range(:)]
# img[200:300, 50:250] = 0,0,255

# [:] select the whole 2D frame
# img[:] = 255,255,255 #Blue Green Red - BGR



# create a line
start_pnt = (20,0) # x,y
end_pnt = (400,400) # x,y
color = (0,255,0) #BGR
thickness = 3

cv2.line(img,start_pnt, end_pnt, color, thickness)



# create a rectangle
start_pnt = (10,20) # x, y
end_pnt = (200,200) # x,y
color = (0,0,255) # BGR
# thickness = cv2.FILLED # fill the rectangle with the color
thickness = 3

cv2.rectangle(img, start_pnt,end_pnt, color, thickness)



#create a circle
center_pnt = (400,50)
radius = 30
color = (0,255,255)
thickness = cv2.FILLED # fill the rectangle with the color
# thickness = 5

cv2.circle(img,center_pnt, radius, color, thickness)




#create a text
text = "OPENCV"
origin_pnt = (200, 300) # x,y
font = cv2.FONT_HERSHEY_COMPLEX
scale = 1.5
color = (0,150,0)
thickness = 2 # must be an integer

cv2.putText(img, text, origin_pnt, font, scale, color, thickness)



cv2.imshow("Image", img)
cv2.waitKey(0)