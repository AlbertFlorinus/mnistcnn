from tkinter.filedialog import askopenfilename
from keras.preprocessing import image
from scipy.ndimage import rotate
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import cv2
import math
from scipy import ndimage

# Creating class for the script
class Run():
	def __init__(self):
		#self.model = load_model("optimum.h5")
		self.model = load_model("ALnet-2.0.h5")
		# printing alnet-0.5 structure 
		for layer in self.model.layers:
			print(layer.output_shape)


	def getBestShift(self, img):
		# preprocessing
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

	def predict_chosen(self):

		# selecting image to predict
		filename = askopenfilename()
		#print(filename)

		# reading image to grayscale
		gray = cv2.imread(filename, 0)

		# inverting color and rezising image to 28x28
		gray = cv2.resize(255-gray, (112, 112), interpolation=cv2.INTER_AREA)

		# (if pixel => 128), pixel = 255

		#(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 3)

		while np.sum(gray[0]) == 0:
		    gray = gray[1:]

		while np.sum(gray[:,0]) == 0:
		    gray = np.delete(gray,0,1)

		while np.sum(gray[-1]) == 0:
		    gray = gray[:-1]

		while np.sum(gray[:,-1]) == 0:
		    gray = np.delete(gray,-1,1)

		rows,cols = gray.shape

		if rows > cols:
		    factor = 80.0/rows
		    rows = 80
		    cols = int(round(cols*factor))
		    gray = cv2.resize(gray, (cols,rows))
		else:
		    factor = 80.0/cols
		    cols = 80
		    rows = int(round(rows*factor))
		    gray = cv2.resize(gray, (cols, rows))

		# get padding
		colsPadding = (int(math.ceil((112-cols)/2.0)),int(math.floor((112-cols)/2.0)))
		rowsPadding = (int(math.ceil((112-rows)/2.0)),int(math.floor((112-rows)/2.0)))

		# apply padding
		gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
		"""def pad_with(vector, pad_width, iaxis, kwargs):
			pad_value = kwargs.get("padder", 10)
			vector[:pad_width[0]] = pad_value
			vector[-pad_width[1]:] = pad_value
			return vector

		gray = np.pad(gray, 16, pad_with)"""

		# some preprocessing 

		shiftx,shifty = self.getBestShift(gray)
		shifted = self.shift(gray,shiftx,shifty)
		img = shifted
		img = gray

		# converting image to array
		img = image.img_to_array(img)

		# reshaping to 1 channel (its only 1 image), 28x28 shape, and 1 color channel (grayscale)
		test_img = img.reshape((1,112,112,1))
		# change data from uint8 to float32
		test_img = img.astype('float32')

		# normalizing pixels
		test_img/=255

		# extend axis at test_img[0] to ready for prediction
		test_img = np.expand_dims(test_img, axis=0)

		# get network output (10 nodes)
		prediction = self.model.predict(test_img)
		print(prediction)

		# reshaping again for showing image to user
		test_img = test_img.reshape((112,112))

		# get highest confidence prediction (our prediction result)
		classname = np.argmax(prediction)
		print(classname)

		plt.gray()
		plt.imshow(test_img)

		# show image with our prediction as title
		plt.title(label=classname)
		plt.show()
		print("Running: "+ "ALnet-2.0.h5")

	def predict_folder(self, pather):

		folder = os.fsencode(pather)
		numberstop = []
		corrects = []
		wrongs = []
		every = []
		imagesk = []

		# predicting every image in selectd folder
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			if filename.endswith((".DS_Store")):
				pass
			else:
				gray = cv2.imread(pather + "/%s"%filename, 0)
				gray = cv2.resize(255-gray, (112, 112), interpolation=cv2.INTER_AREA)
				#(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
				gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)

				
				while np.sum(gray[0]) == 0:
				    gray = gray[1:]

				while np.sum(gray[:,0]) == 0:
				    gray = np.delete(gray,0,1)

				while np.sum(gray[-1]) == 0:
				    gray = gray[:-1]

				while np.sum(gray[:,-1]) == 0:
				    gray = np.delete(gray,-1,1)

				rows,cols = gray.shape

				if rows > cols:
				    factor = 80.0/rows
				    rows = 80
				    cols = int(round(cols*factor))
				    gray = cv2.resize(gray, (cols,rows))
				else:
				    factor = 80.0/cols
				    cols = 80
				    rows = int(round(rows*factor))
				    gray = cv2.resize(gray, (cols, rows))

				#get padding
				colsPadding = (int(math.ceil((112-cols)/2.0)),int(math.floor((112-cols)/2.0)))
				rowsPadding = (int(math.ceil((112-rows)/2.0)),int(math.floor((112-rows)/2.0)))
				#apply apdding
				gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
				shiftx,shifty = self.getBestShift(gray)
				shifted = self.shift(gray,shiftx,shifty)
				gray = shifted

				img = gray
				img = image.img_to_array(img)
				test_img = img.reshape((1,112,112,1))
				test_img = img.astype('float32')
				test_img/=255
				test_img = np.expand_dims(test_img, axis=0)
				img_class = self.model.predict(test_img)
				prediction = img_class
				classname = np.argmax(prediction)
				every.append(filename)
				if filename.startswith(str(classname)):
					corrects.append(filename)
				else:
					wrongs.append(filename)

				test_img = test_img.reshape((112,112))
				imagesk.append(test_img)

		c = 0
		w = 0
		e = 0
		for i in every:
			e += 1

		for i in corrects:
			c += 1

		for i in wrongs:
			w += 1
		print("all numbers")
		print(e)
		print(every)
		print("right guess")
		print(c)
		print(corrects)
		print("wrong guess")
		print(w)
		print(wrongs)
		print("Running: "+ "ALnet-2.0.h5")



