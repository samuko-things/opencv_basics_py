# -*- coding: utf-8 -*-

import os
import traceback
import logging

import cv2
from skimage.feature import hog
import joblib
from common.config import get_config
from common.image_transformation import resize_image

import time


logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)


def get_image_from_label(label):
    testing_images_dir_path = get_config('testing_images_dir_path')
    image_path = os.path.join(testing_images_dir_path, label, '001.jpg')
    image = cv2.imread(image_path)
    return image


def main():

    camera = cv2.VideoCapture(0)
    while True:
        start_time = time.time()
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to capture image!")
            continue

        cv2.imshow("Webcam recording", frame)
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(frame,(128,128))
        try:
            hog_feature, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), block_norm= 'L2',visualize=True)
            classifier_model = joblib.load("lane.pkl")
            predicted_labels = classifier_model.predict([hog_feature])
            predicted_label = predicted_labels[0]

            # logger.info("Predicted label = {}".format(predicted_label))
            # predicted_image = get_image_from_label(predicted_label)
            # predicted_image = resize_image(predicted_image, 200)
            # cv2.imshow("Prediction = '{}'".format(predicted_label), predicted_image)
            print('Predicted: {} '.format(predicted_label), round((time.time()-start_time),3))
        except Exception:
            exception_traceback = traceback.format_exc()
            logger.error("Error applying image transformation")
            logger.debug(exception_traceback)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.waitKey(100)
#        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    logger.info("The program completed successfully !!")


if __name__ == '__main__':
    main()
