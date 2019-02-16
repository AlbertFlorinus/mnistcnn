import os
from scipy.ndimage import rotate
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, Adadelta
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import cv2
import math
from scipy import ndimage
import skimage.measure

def model1(predict, pather):

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train/=255
	X_test/=255
	number_of_classes = 10
	Y_train = np_utils.to_categorical(y_train, number_of_classes)
	Y_test = np_utils.to_categorical(y_test, number_of_classes)
	model = Sequential()
	model.add(Conv2D(32, (5,5), activation="relu", input_shape=(28,28,1)))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(64, (5,5), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(1024, activation="relu"))
	model.add(Dense(400, activation="relu"))
	model.add(Dense(number_of_classes, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
	model.load_weights("trawe/model1.h5")
	score = model.evaluate(X_test, Y_test, verbose=0)
	print("Test loss: ", score[0])
	print("Test accuracy: ", score[1])
	for layer in model.layers:
		print(layer.output_shape)

	if predict == True:
		def getBestShift(img):
		    cy,cx = ndimage.measurements.center_of_mass(img)

		    rows,cols = img.shape
		    shiftx = np.round(cols/2.0-cx).astype(int)
		    shifty = np.round(rows/2.0-cy).astype(int)

		    return shiftx,shifty

		def shift(img,sx,sy):
		    rows,cols = img.shape
		    M = np.float32([[1,0,sx],[0,1,sy]])
		    shifted = cv2.warpAffine(img,M,(cols,rows))
		    return shifted


		folder = os.fsencode(pather)
		#corrects = []
		#wrongs = []
		#every = []
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			#print(filename)
			if filename.endswith((".DS_Store")):
				pass
			else:
				gray = cv2.imread("siffror/"+str(filename), 0)
				gray = cv2.resize(255-gray, (28, 28))
				(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

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
				    factor = 20.0/rows
				    rows = 20
				    cols = int(round(cols*factor))
				    gray = cv2.resize(gray, (cols,rows))
				else:
				    factor = 20.0/cols
				    cols = 20
				    rows = int(round(rows*factor))
				    gray = cv2.resize(gray, (cols, rows))

				colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
				rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
				gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

				shiftx,shifty = getBestShift(gray)
				shifted = shift(gray,shiftx,shifty)
				gray = shifted

				#cv2.imwrite("pythonapp/AI/siffror/7IMGtres.PNG", gray)
				#cv2.imwrite("7IMGNEW.PNG", gray)
				cv2.imwrite("pro-img/"+str(filename), gray)

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def preprocessor(pather, writeto):
		folder = os.fsencode(pather)
		os.mkdir(writeto)
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			if filename.endswith((".DS_Store")):
				pass
			else:
				#gray = cv2.imread("siffror/"+str(filename), 0)
				gray = cv2.imread(pather + "/%s"%filename, 0)
				gray = cv2.resize(255-gray, (28, 28))
				(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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
				    factor = 20.0/rows
				    rows = 20
				    cols = int(round(cols*factor))
				    gray = cv2.resize(gray, (cols,rows))
				else:
				    factor = 20.0/cols
				    cols = 20
				    rows = int(round(rows*factor))
				    gray = cv2.resize(gray, (cols, rows))

				colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
				rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
				gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

				shiftx,shifty = getBestShift(gray)
				shifted = shift(gray,shiftx,shifty)
				gray = shifted

				cv2.imwrite(writeto + "/%s"%filename, gray)

def preprocessormax(pather, writeto):
		folder = os.fsencode(pather)
		os.mkdir(writeto)
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			if filename.endswith((".DS_Store")):
				pass
			else:
				#gray = cv2.imread("siffror/"+str(filename), 0)
				gray = cv2.imread(pather + "/%s"%filename, 0)
				gray = cv2.resize(255-gray, (112, 112))
				gray = skimage.measure.block_reduce(gray, (2,2), np.max)
				gray = cv2.resize(gray, (28, 28))
				(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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
				    factor = 20.0/rows
				    rows = 20
				    cols = int(round(cols*factor))
				    gray = cv2.resize(gray, (cols,rows))
				else:
				    factor = 20.0/cols
				    cols = 20
				    rows = int(round(rows*factor))
				    gray = cv2.resize(gray, (cols, rows))

				colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
				rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
				gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

				shiftx,shifty = getBestShift(gray)
				shifted = shift(gray,shiftx,shifty)
				gray = shifted

				cv2.imwrite(writeto + "/%s"%filename, gray)
#model1(predict=True, pather="siffror")
#preprocessor("digits/siffrorsvåra", "maxpooleddigitssvåra")
#preprocessormax("digits/siffrorsvåra", "digits/maxpooleddigitssvåra")

