import os
from scipy.ndimage import rotate
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

def simplecnnmodel(predict, pather):

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
	model.add(Dense(1000, activation="relu"))
	model.add(Dense(400, activation="relu"))
	model.add(Dense(number_of_classes, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
	model.load_weights("trawe/simplestAdam-cnn.h5")
	score = model.evaluate(X_test, Y_test, verbose=0)
	print("Test loss: ", score[0])
	print("Test accuracy: ", score[1])
	for layer in model.layers:
		print(layer.output_shape)
	if predict == True:
		folder = os.fsencode(pather)
		corrects = []
		wrongs = []
		every = []
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			if filename.endswith((".DS_Store")):
				pass
			else:
				img = image.load_img(path=pather + "/%s"%filename,color_mode="grayscale",target_size=(28,28))
				img = image.img_to_array(img)
				test_img = img.reshape((1,28,28,1))
				test_img = img.astype('float32')
				test_img/=255
				test_img = np.expand_dims(test_img, axis=0)
				img_class = model.predict(test_img)
				prediction = img_class
				classname = np.argmax(prediction)
				every.append(filename)
				if filename.startswith(str(classname)):
					corrects.append(filename)
				else:
					wrongs.append(filename)
		every.sort()
		corrects.sort()
		wrongs.sort()
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
		print("Running: "+ "modelsimple.h5")

def simplecnnmodeldrop(predict, pather):

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
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(1024, activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(400, activation="relu"))
	model.add(Dense(number_of_classes, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
	model.load_weights("trawe/simplestAdamdrop-cnn.h5")
	score = model.evaluate(X_test, Y_test, verbose=0)
	print("Test loss: ", score[0])
	print("Test accuracy: ", score[1])
	for layer in model.layers:
		print(layer.output_shape)
	if predict == True:
		folder = os.fsencode(pather)
		corrects = []
		wrongs = []
		every = []
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			if filename.endswith((".DS_Store")):
				pass
			else:
				img = image.load_img(path=pather + "/%s"%filename,color_mode="grayscale",target_size=(28,28))
				img = image.img_to_array(img)
				test_img = img.reshape((1,28,28,1))
				test_img = img.astype('float32')
				test_img/=255
				test_img = np.expand_dims(test_img, axis=0)
				img_class = model.predict(test_img)
				prediction = img_class
				classname = np.argmax(prediction)
				every.append(filename)
				if filename.startswith(str(classname)):
					corrects.append(filename)
				else:
					wrongs.append(filename)
		every.sort()
		corrects.sort()
		wrongs.sort()
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
		print("Running: "+ "modelsimpledrop.h5")

def inprogcnn(predict, pather):

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
	model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64,(3, 3)))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	model.load_weights("trawe/model-cnn9.h5")
	score = model.evaluate(X_test, Y_test, verbose=0)
	print("Test loss: ", score[0])
	print("Test accuracy: ", score[1])
	if predict == True:
		folder = os.fsencode(pather)
		corrects = []
		wrongs = []
		every = []
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			if filename.endswith((".DS_Store")):
				pass
			else:
				img = image.load_img(path=pather + "/%s"%filename,color_mode="grayscale",target_size=(28,28))
				img = image.img_to_array(img)
				test_img = img.reshape((1,28,28,1))
				test_img = img.astype('float32')
				test_img/=255
				test_img = np.expand_dims(test_img, axis=0)
				img_class = model.predict(test_img)
				prediction = img_class
				classname = np.argmax(prediction)
				every.append(filename)
				if filename.startswith(str(classname)):
					corrects.append(filename)
				else:
					wrongs.append(filename)
		every.sort()
		corrects.sort()
		wrongs.sort()
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
		print("Running: "+ "modeldeep.h5")

def kerasex(predict, pather):

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
	model.add(Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)))
	model.add(Conv2D(64, (3,3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(number_of_classes, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=["accuracy"])
	model.load_weights("trawe/kerasex.h5")
	score = model.evaluate(X_test, Y_test, verbose=0)
	print("Test loss: ", score[0])
	print("Test accuracy: ", score[1])
	if predict == True:
		folder = os.fsencode(pather)
		corrects = []
		wrongs = []
		every = []
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			if filename.endswith((".DS_Store")):
				pass
			else:
				img = image.load_img(path=pather + "/%s"%filename,color_mode="grayscale",target_size=(28,28))
				img = image.img_to_array(img)
				test_img = img.reshape((1,28,28,1))
				test_img = img.astype('float32')
				test_img/=255
				test_img = np.expand_dims(test_img, axis=0)
				img_class = model.predict(test_img)
				prediction = img_class
				classname = np.argmax(prediction)
				every.append(filename)
				if filename.startswith(str(classname)):
					corrects.append(filename)
				else:
					wrongs.append(filename)
		every.sort()
		corrects.sort()
		wrongs.sort()
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
		print("Running: "+ "kerasex5.h5")

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
	if predict == True:
		folder = os.fsencode(pather)
		corrects = []
		wrongs = []
		every = []
		for file in os.listdir(folder):
			filename = os.fsdecode(file)
			if filename.endswith((".DS_Store")):
				pass
			else:
				img = image.load_img(path=pather + "/%s"%filename,color_mode="grayscale",target_size=(28,28))
				img = image.img_to_array(img)
				test_img = img.reshape((1,28,28,1))
				test_img = img.astype('float32')
				test_img/=255
				test_img = np.expand_dims(test_img, axis=0)
				img_class = model.predict(test_img)
				prediction = img_class
				classname = np.argmax(prediction)
				every.append(filename)
				if filename.startswith(str(classname)):
					corrects.append(filename)
				else:
					wrongs.append(filename)
		every.sort()
		corrects.sort()
		wrongs.sort()
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
		print("Running: "+ "model1.h5")

def model1ui():

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

	filename = askopenfilename()
	img = image.load_img(path=filename, color_mode="grayscale",target_size=(28,28))
	if filename.endswith((".JPG")):
		img = rotate(img, 270)
	img = image.img_to_array(img)
	test_img = img.reshape((1,28,28,1))
	test_img = img.astype('float32')
	test_img/=255
	test_img = np.expand_dims(test_img, axis=0)
	img_class = model.predict(test_img)
	test_img = test_img.reshape((28,28))
	prediction = img_class
	classname = np.argmax(prediction)
	plt.gray()
	plt.imshow(test_img)
	plt.title(label=classname)
	plt.show()
	print("Running: "+ "model1.h5")
