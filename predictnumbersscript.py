import os
from scipy.ndimage import rotate
import numpy as np
#np.random.seed(1)
from tensorflow import set_random_seed
#set_random_seed(2)
import matplotlib.pyplot as plt
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
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.load_weights("model-cnn6.h5")
path = "siffror"
folder = os.fsencode(path)
corrects = []
wrongs = []
every = []
for file in os.listdir(folder):
	filename = os.fsdecode(file)
	if filename.endswith((".DS_Store")):
		pass
	else:
		img = image.load_img(path="siffror/%s"%filename,color_mode="grayscale",target_size=(28,28))
		if filename.endswith(('.JPG')): # whatever file types you're using...
			img = rotate(img, 270)
		img = image.img_to_array(img)
		img = 255-img
		img[np.where((img==[255.]).all(axis=2))] = [40.]
		img[np.where((img < [100.]).all(axis=2))] = [20.]
		test_img = img.reshape((1,28,28,1))
		test_img = img.astype('float32')
		test_img/=255
		test_img = np.expand_dims(test_img, axis=0)
		img_class = model.predict(test_img, batch_size=1, verbose=1)
		prediction = img_class
		print(prediction)
		classname = np.argmax(prediction)
		print(filename)
		print("Class: ",classname)
		every.append(filename)
		if filename.startswith(str(classname)):
			#print("yes")
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

