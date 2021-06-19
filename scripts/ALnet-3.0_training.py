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
from keras.callbacks import LearningRateScheduler, CSVLogger
from cv2 import cv2


# This is the script used for designing and training ALnet-2.0

def alnet():

	"""
	Training results:
	val_loss: 0.0157 - val_acc: 0.9950
	Test loss:  0.01574255358175287
	Test accuracy:  0.995
	[Finished in 2489.3s]
	"""
	
	# gathering the mnist-dataset, train is used for training,
	# test is used to predict images previously unseen,
	# we do this to ensure overfitting has not occurred,
	# y are labels to X which are the images

	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	
	# reshaping for keras compatibility
	X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

	# changing datatype from uint8 to float32,
	# this is to allow for normalizising the pixel value,
	# between 0 and 1
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# normalizing pixels
	X_train/=255
	X_test/=255

	# one-hot-encoding the outputs, or in simpler terms,
	# storing the outputs as a vector of 1x10
	number_of_classes = 10
	Y_train = np_utils.to_categorical(Y_train, number_of_classes)
	Y_test = np_utils.to_categorical(Y_test, number_of_classes)

	# Three steps to create a CNN
	# 1. Convolution
	# 2. Activation
	# 3. Pooling
	# Repeat Steps 1,2,3 for adding more hidden layers

	# 4. After that make a fully connected network
	# This fully connected network gives ability to the CNN
	# to classify the samples

	# creating a sequential model in keras, linear stack of layers
	# code below is the ALnet-2.0 model design
	model = Sequential()

	# each convolutional/conv layer distorts the input arrays
	# each model.add creates a new layer in the network
	# adding a spatial convolutional/conv layer and 
	# declaring input shape required for the img array
	# input shape value only needed for first conv
	# 32 for the amount of filters and thus also feature maps outputed,
	# (3,3) for filter size, default stride is 1

	model.add(Conv2D(32, (3,3), activation="relu", input_shape = (28,28,1), padding = "same"))
	
	# adding a batchnormalization layer to reduce a batchs covariant shift,
	# normalizing the images to execute more effectively,
	model.add(BatchNormalization())

	model.add(Conv2D(32, (3,3), activation="relu"))
	model.add(BatchNormalization())

	# adding conv layer with 5x5 filter and a stride of 2 instead of max pooling,
	# downsampling image but retaining import data for classification.
	model.add(Conv2D(32, (5,5), strides=2, padding="same",activation="relu"))
	model.add(BatchNormalization())

	# using dropout with a rate 0f 0.4, this randomly "drops",
	# 40% of the nodes to a output value of 0 each iteration, which helps prevent overfitting
	model.add(Dropout(0.4))

	# raise amount from 32 to 64
	model.add(Conv2D(64, (3,3), activation="relu"))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), activation="relu"))
	model.add(BatchNormalization())


	model.add(Conv2D(64, (5,5), strides=2, padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	# flattening the input to a 1d array,
	# flattening the pixel data of 64 4x4 arrays
	# to a 1d array containing 1024 pixel values,
	# not 1024 pixels of the original image but of the,
	# outputs from the convolutional neural network
	# this ends the spatial/convolutional part of the network
	model.add(Flatten())

	# adding a fully connected layer of 128 neurons
	model.add(Dense(128, activation="relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	# Final layer of 10 neurons with a softmax activation,
	# this outputs a prediction (number with highest activation value)
	model.add(Dense(number_of_classes, activation="softmax"))

	# Declaring loss function and optimizer,
	# Adam is an enhancement of SGD.
	# Accuracy metric for us to get results for evaluating the model 
	model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

	# Number of images to iterate simultaneously before each weight update
	batchsize = 64

	# Augmentning training data for better generalisation,
	# and prevent overfitting
	gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.15)
	train_generator = gen.flow(X_train, Y_train, batch_size=batchsize)

	# Reducing learning rate to 95% of the last epoch,
	# speeding up convergence by keeping weight updates smaller as the model,
	# approaches convergence.
	annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

	# Log file for tracking information about the learning process and its metrics
	csv_logger = CSVLogger("training_test.log", append=True, separator=";")
	
	# starting training, validation_data is mnist data not trained on,
	# to ensure us we arent overfitting to the training set but actually generalising
	model.fit_generator(train_generator,steps_per_epoch=X_train.shape[0]//batchsize, epochs=10, 
                    validation_data=(X_test, Y_test), callbacks=[annealer, csv_logger], verbose=1)
	history = model.fit(train_generator,steps_per_epoch=X_train.shape[0]//batchsize, epochs=10, 
                    validation_data=(X_test, Y_test), callbacks=[annealer, csv_logger], verbose=1)


	model.save("ALnet-3.0.h5")
	score = model.evaluate(X_test, Y_test, verbose=1)
	print("Test loss: ", score[0])
	print("Test accuracy: ", score[1])

if __name__ == "__main__":
	alnet()
