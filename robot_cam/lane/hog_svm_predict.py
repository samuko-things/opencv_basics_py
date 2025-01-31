# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:37:51 2021

@author: Olaniyi Taofeeq O
"""

import numpy as np
import cv2
import os
import time
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib
from skimage.feature import hog
from sklearn.metrics import classification_report
import logging
import os
# dictionary of labels


logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)


def datagen(img_folder):
    """
    Function: datagen 
    
    Input: 
        List of filenames with their absolute paths
    
    Output: Train data and labels depending on mode value
    
    Description: This function computes HOG features for each image in the data/image/train folder, assigns label to the descriptor vector of the image and returns the final train/test data and labels matrices used for feeding the SVM in training phase or predicting the label of test data.
    
    """

    data = []
    label = []


    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            filename = os.path.join(img_folder, dir1,  file)

            # read image
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(128,128))
            #image = Standardizer.read_image(filename)
    
            # compute HOG features
            
            # = FeatureExtractor()
            otsu_threshold, image_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
            #hog_feature, spatial, color, canny = featureExtractor.extract_features(image)
            hog_feature, hog_image = hog(image_result, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), block_norm= 'L2',visualize=True)
    
            #image_path = os.path.join(file, filename)
            image_label = os.path.splitext(os.path.basename(dir1))[0]
#            image_info = {}
#            image_info['image_path'] = image_path
#            image_info['image_label'] = image_label
            #images_labels_list.append(image_info)
            
           
         
            
            classifier_model = joblib.load("lane.pkl")
            prediction = classifier_model.predict([hog_feature])
            predicted_label = prediction[0]
            
           
            print('"Image label: {}" Predicted: {} '.format(image_label,
                        predicted_label,))
            
            
            
            #print(image_label)
    
            #append descriptor and label to train/test data, labels
            data.append(hog_feature)
            label.append(image_label)
           

    # return data and label
    return data, label

def main():
    # list of training and test files
    test_folder = r"lane/data/images/test/"
    #testing data & labels
    data, label = datagen(test_folder)
  
    print("Predicting from test folder")

    # testing phase: SVM , predict model to testing data ------------------------------
    model = joblib.load("lane.pkl")

    # predict labels for test data
    predictions = model.predict(data)
    
    # compute accuracy
    accuracy = accuracy_score(label, predictions) * 100
    print("\nAccuracy: %.2f" % accuracy + "%")
    
    


if __name__ == "__main__": 
    start_time = time.time()
    main()
    print('Execution time: %.2f' % (time.time() - start_time) + ' seconds\n')