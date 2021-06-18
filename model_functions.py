from tkinter.filedialog import askopenfilename
from keras.preprocessing import image
from scipy.ndimage import rotate
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from cv2 import cv2


import math
from scipy import ndimage

import sys

# Creating class for the script
class Run():
	def __init__(self):
		self.model = load_model("ALnet-2.0.h5")
		print("\nStructure of Alnet-2.0")
		# printing Alnet-2.0 structure 
		for layer in self.model.layers:
			print(layer.output_shape)

	def getBestShift(self, img):
		# preprocessing

		# calculate the center of mass of the image along y and x axis
	    cy,cx = ndimage.measurements.center_of_mass(img)
	    
	    rows,cols = img.shape
	    shiftx = np.round(cols/2.0-cx).astype(int)
	    shifty = np.round(rows/2.0-cy).astype(int)
	    return shiftx,shifty

	def shift(self, img,sx,sy):
		# preprocessing
	    rows,cols = img.shape
	    M = np.float32([[1,0,sx],[0,1,sy]])
	    shifted = cv2.warpAffine(img,M,(cols,rows))
	    return shifted

	def image_manip(self, subj):

		# this method uses the methods "shift" and "getBestShift" to preprocess the image.

		# this code (50-62) removes the rows and columns at the edge of the image,
		# which are completely black as a result of previous thresholding.
		# this resize the image to 71% rounded up of the previous size,
		# so that a size of 112x112 now becomes 80x80, with the edges not containing information,
		# of the digits now removed.
		while np.sum(subj[0]) == 0:
		    subj = subj[1:]

		while np.sum(subj[:,0]) == 0:
		    subj = np.delete(subj,0,1)

		while np.sum(subj[-1]) == 0:
		    subj = subj[:-1]

		while np.sum(subj[:,-1]) == 0:
		    subj = np.delete(subj,-1,1)

		rows,cols = subj.shape

		# this resize the
		if rows > cols:
		    factor = 80.0/rows
		    rows = 80
		    cols = int(round(cols*factor))
		    subj = cv2.resize(subj, (cols,rows))
		else:
		    factor = 80.0/cols
		    cols = 80
		    rows = int(round(rows*factor))
		    subj = cv2.resize(subj, (cols, rows))

		# get padding
		colsPadding = (int(math.ceil((112-cols)/2.0)),int(math.floor((112-cols)/2.0)))
		rowsPadding = (int(math.ceil((112-rows)/2.0)),int(math.floor((112-rows)/2.0)))

		# apply padding
		subj = np.lib.pad(subj,(rowsPadding,colsPadding),'constant')

		# some preprocessing 
		shiftx,shifty = self.getBestShift(subj)
		shifted = self.shift(subj,shiftx,shifty)
		subj_done = shifted
		return subj_done

	def predict_chosen(self):

		print("")
		# selecting image to predict
		filename = askopenfilename()

		# reading image to grayscale
		gray = cv2.imread(filename, 0)

		# dilate img
		dilated_gray = cv2.dilate(gray, np.ones((7,7), np.uint8))

		# median blur the dilated img to further suppress detail
		bg_gray = cv2.medianBlur(dilated_gray, 21)

		# calculate difference between gray(original img) and bg_gray.
		# identical pixels will be black(close to 0 difference), the digits details will be white(large difference).
		diff_gray = 255 - cv2.absdiff(gray, bg_gray)

		# normalize the image, so that we use the full dynamic range.
		norm_gray = diff_gray.copy()
		cv2.normalize(diff_gray, norm_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

		# now the image is still gray in some areas, so we truncate that away and re-normalize the image.
		_, thr_gray = cv2.threshold(norm_gray, 253, 0, cv2.THRESH_TRUNC)
		cv2.normalize(thr_gray, thr_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

		# inverting color and rezising image to 112x112
		gray1 = cv2.resize(255-thr_gray, (112, 112), interpolation=cv2.INTER_AREA)

		# we have variables gray and gray2 to differentiate between two ways of thresholding.
		# Alnet-2.0 will make predictions on both of these, and combine the two to get our resulting classification.
		gray = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)
		(thresh, gray2) = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		
		img = self.image_manip(gray)
		img2 = self.image_manip(gray2)

		# converting PIL image to numpy array
		img = image.img_to_array(img)
		img2 = image.img_to_array(img2)

		# reshaping to 1 channel (its only 1 image), 112x112 shape, and 1 color channel (grayscale)
		test_img = img.reshape((1,112,112,1))
		test_img2 = img2.reshape((1,112,112,1))

		# change datatype from uint8 to float32
		test_img = img.astype('float32')
		test_img2 = img2.astype('float32')

		# normalizing pixels in the decimal range 0 to 1
		test_img /= 255
		test_img2 /= 255

		# extend axis at index[0] to ready for prediction
		test_img = np.expand_dims(test_img, axis=0)
		test_img2 = np.expand_dims(test_img2, axis=0)

		# get network output (10 nodes)
		# prediction_list is prediction of gray
		# prediction2_list is prediction of gray2
		prediction_list = self.model.predict(test_img)
		prediction2_list = self.model.predict(test_img2)

		# x1 and x2 are indexes for elements (what digit is guessed) of highest value for images gray and gray2
		x1 = np.argmax(prediction_list)
		x2 = np.argmax(prediction2_list)

		# c is the highest value in prediction_list
		# v is the highest value in prediction2_list
		c = prediction_list[0][x1]
		v = prediction2_list[0][x2]

		# b is the value of prediction_list at index of prediction2_list highest value
		# n is the value of prediction2_list at index of predition_list highest value
		b = prediction_list[0][x2]
		n = prediction2_list[0][x1]

		# this helps us decide what prediction is most likely to be correct, by comparing the values below.
		k = (c+b)/2
		j = (v+n)/2
		if k > j:
			classname = x1

		if j > k:
			classname = x2

		# reshaping again for showing image to user
		test_img = test_img.reshape((112,112))
		test_img2 = test_img2.reshape((112,112))

		# lines 181-195 displays to the user gray and gray2, values of k and j, and the estimated classname for the original image chosen
		fig = plt.figure(figsize=(8,8))
		columns = 2
		rows = 1
		plt.gray()
		fig.add_subplot(rows, columns, 1).set_title("adaptive_Threshold\n k = {}".format(k))
		plt.imshow(test_img)
		fig.add_subplot(rows, columns, 2).set_title("Threshold\n j = {}".format(j))
		plt.imshow(test_img2)
		if k > j:
			fig.suptitle("Image shows a {}\n k > j".format(classname))

		if j > k:
			fig.suptitle("Image shows a {}\n j > k".format(classname))

		plt.show()

		print("\nadaptive_Threshold prediction:\n",prediction_list,"\n")
		print("Threshold prediction:\n",prediction2_list)
		print("\nRunning: "+ "Alnet-2.0.")

	def predict_folder(self, pather):
		
		dirs = os.listdir(pather)
		corrects = []
		wrongs = []
		every = []

		for folders in dirs:
			fold = os.fsdecode(folders)
			if fold.endswith((".DS_Store")):
				pass
			else:
				for filename in os.listdir(pather + "/" + fold):
					if filename.endswith((".DS_Store")):
						pass
					else:
						gray = cv2.imread(pather + "/%s"%fold + "/%s"%filename, 0)
						dilated_gray = cv2.dilate(gray, np.ones((7,7), np.uint8))
						bg_gray = cv2.medianBlur(dilated_gray, 21)
						diff_gray = 255 - cv2.absdiff(gray, bg_gray)
						norm_gray = diff_gray.copy()
						cv2.normalize(diff_gray, norm_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

						_, thr_gray = cv2.threshold(norm_gray, 253, 0, cv2.THRESH_TRUNC)
						cv2.normalize(thr_gray, thr_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
						
						gray1 = cv2.resize(255-thr_gray, (112, 112), interpolation=cv2.INTER_AREA)

						gray = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)
						(thresh, gray2) = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
						
						img = self.image_manip(gray)
						img2 = self.image_manip(gray2)

						img = image.img_to_array(img)
						img2 = image.img_to_array(img2)

						test_img = img.reshape((1,112,112,1))
						test_img2 = img2.reshape((1,112,112,1))

						test_img = img.astype('float32')
						test_img2 = img2.astype('float32')

						test_img /= 255
						test_img2 /= 255

						test_img = np.expand_dims(test_img, axis=0)
						test_img2 = np.expand_dims(test_img2, axis=0)

						prediction = self.model.predict(test_img)
						prediction2 = self.model.predict(test_img2)

						x1 = np.argmax(prediction)
						x2 = np.argmax(prediction2)

						c = prediction[0][x1]
						v = prediction2[0][x2]

						b = prediction[0][x2]
						n = prediction2[0][x1]

						k = (c+b)/2
						j = (v+n)/2
						if k > j:
							classname = x1
							
						if j > k:
							classname = x2
							
						print(filename+ "prediction: ", classname)

						# This list is to keep count of how many images are being classified
						every.append(filename)

						# This is for benchmark tests during development and demonstrating larger sets of images,
						# This requires the images to be labeled to see how many predictions were right or wrong,
						# THIS IS OPTIONAL, if you dont label your images you wont get a list of prediction accuracy,
						# But the predictions will be printed as outputs next to each image filename.
						if filename.startswith(str(classname)):
							corrects.append(filename)
						else:
							wrongs.append(filename)

		co = 0
		w = 0
		e = 0
		for i in every:
			e += 1

		for i in corrects:
			co += 1

		for i in wrongs:
			w += 1
		print("all numbers")
		print(e)
		print("right guess")
		print(co)
		print("wrong guess")
		print(w)
		print(wrongs)
		print("Running: "+ "ALnet-2.0")




