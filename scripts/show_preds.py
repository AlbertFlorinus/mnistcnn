import os
import numpy as np
import keras

from keras.preprocessing import image
from cv2 import cv2


import matplotlib.pyplot as plt

#THIS IS WIP, dont use

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

def preprocessor2(img):
    gray = img
    dilated_gray = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg_gray = cv2.medianBlur(dilated_gray, 21)
    diff_gray = 255-cv2.absdiff(gray, bg_gray)
    norm_gray = diff_gray.copy()

    _, thr_gray = cv2.threshold(norm_gray, 253, 0, cv2.THRESH_TRUNC)

    norm_img = np.zeros((320,240))
    norm_gray = cv2.normalize(diff_gray, norm_img, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    test_img = diff_gray - norm_gray  
    test_img2 = cv2.dilate(test_img, np.ones((5,5)))

    ret2, gray3 = cv2.threshold(test_img2,40,255,cv2.THRESH_OTSU)

    gray3 = cv2.resize(gray3, (28,28), interpolation = cv2.INTER_AREA)

    ret3, gray4 = cv2.threshold(test_img2, 40, 255, cv2.THRESH_TRUNC)

    gray4 = cv2.resize(gray4, (28,28), interpolation = cv2.INTER_AREA)

    gray5 = cv2.resize(thr_gray, (28,28), interpolation=cv2.INTER_AREA)

    _, gray6 = cv2.threshold(norm_gray, 30, 30, cv2.THRESH_OTSU)
    #gray6 = cv2.resize(gray6, (28,28), interpolation=cv2.INTER_AREA)
    #gray6 = cv2.resize(thr_gray, (28,28), interpolation=cv2.INTER_AREA)

    #(ret6, gray6) = cv2.threshold(test_img2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #gray6 = cv2.resize(norm_gray, (28,28), interpolation=cv2.INTER_AREA)
    #gray = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)
    #(thresh, gray2) = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    

    return gray3, gray4, gray5, gray6

def into_alnet(img):
    img = img.reshape((1,28,28,1))
    img = img.astype('float32')
    img /= 255
    return img

if __name__ == "__main__":

    from tkinter.filedialog import askopenfilename
    filename = askopenfilename()
    img_orig = cv2.imread(filename, 0)
    gray3, gray4, gray5, gray6 = preprocessor2(img_orig)
    plt.gray()
    f, axarr = plt.subplots(nrows=1,ncols=4)
    plt.sca(axarr[0]); 
    plt.imshow(gray3); plt.title('0,255,OTSU')
    plt.sca(axarr[1]); 
    plt.imshow(gray4); plt.title('128,255,OTSU')
    plt.sca(axarr[2]); 
    plt.imshow(gray5); plt.title('NONE')
    plt.sca(axarr[3]); 
    plt.imshow(gray6); plt.title('')
    plt.show()
