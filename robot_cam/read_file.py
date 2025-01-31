import cv2 # OpenCV library
import time

import numpy as np



if __name__=='__main__':
    content = np.loadtxt('data1.txt')
    print("\nContent in file1.txt:\n")
    print(content[:-1,:])
    print(content.shape)