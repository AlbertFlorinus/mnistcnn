import os
import numpy as np
import keras
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from cv2 import cv2

def preprocessor(img):
    gray = img
    dilated_gray = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg_gray = cv2.medianBlur(dilated_gray, 21)
    diff_gray = 255-cv2.absdiff(gray, bg_gray)
    norm_gray = diff_gray.copy()
    norm_img = np.zeros((320,240))
    norm_gray = cv2.normalize(diff_gray, norm_img, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    test_img = diff_gray - norm_gray  
    test_img2 = cv2.dilate(test_img, np.ones((5,5)))

    ret2,gray3 = cv2.threshold(test_img2,0,255,cv2.THRESH_OTSU)
    gray3 = cv2.resize(gray3, (28,28), interpolation = cv2.INTER_AREA)
    return gray3, gray3.copy()

def into_alnet(img):
    img = img.reshape((1,28,28,1))
    img = img.astype('float32')
    img /= 255
    return img