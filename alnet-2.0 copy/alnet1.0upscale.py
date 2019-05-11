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
from keras.callbacks import LearningRateScheduler
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt

(X_traine, y_train), (X_teste, y_test) = mnist.load_data()

X_train = np.empty((60000,56,56))
X_test = np.empty((10000,56,56))
for i in range(60000):
	imgs = X_traine[i]
	enlargeder = cv2.resize(imgs, (56, 56), interpolation=cv2.INTER_AREA)
	X_train[i,:,:] = enlargeder

for i in range(10000):
	imgs = X_teste[i]
	enlargeder = cv2.resize(imgs, (56, 56), interpolation=cv2.INTER_AREA)
	X_test[i,:,:] = enlargeder

X_train = X_train.reshape(X_train.shape[0], 56, 56, 1)
X_test = X_test.reshape(X_test.shape[0], 56, 56, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing pixels
X_train/=255
X_test/=255

# one-hot-encoding the outputs, or in simpler terms,
# storing the outputs as a vector of 1x10
number_of_classes = 10
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
"""print(hj.shape)
plt.gray()
plt.imshow(hj[0,:,:])
plt.show()"""

"""img = X_train[0]
enlarged = cv2.resize(img, (56, 56), interpolation=cv2.INTER_AREA)
plt.gray()
plt.imshow(enlarged)
plt.show()"""
#X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#print(X_train[0].shape)
#print(X_train.shape)
#print(X_train[0])
#print(X_train[0])
#print(X_train.shape)
"""c = 0
for i in range(len(X_train)):
	print(i)
	print(X_train[i])
	c += 1
	if c > 2:
		break
	#if c == 1:
		#print(X_train[i])
print(c)"""

#print(c)
#print(X_train[0])

#X_train1 = X_train.copy().ravel()
#y_train1 = y_train.copy().ravel()

#X_train2 = np.resize(X_train1, 56*56*60000)
#y_train2 = np.resize(y_train1, 56*56*60000)
#X_train1 = X_train.copy().ravel()

#X_train = X_train.copy().ravel()
"""c = 0
k = 0
for i in range(len(X_train)):
	c += 1
	if c == 784:
		#tempe = X_train[:784]
		#X_train[k] = X_train[:784].reshape(28,28)
		X_train[k] = np.reshape(X_train[:784], (28,28))
		k += 1



for i in range(len(X_train)):
	#for j in range(len(X_train1[i])):
	X_train[i] = X_train[i].resize(56*56)
	k += 1
	if k == 10000:
		print(X_train[i].shape)

l = 0
for i in range(len(X_train)):
	l += 1
	#for j in range(len(X_train1[i])):
print(l)"""
#print(X_train.shape)
#X_train = X_train.reshape(X_train.shape[0], 56, 56, 1)
#X_train[0] = cv2.resize(X_train[0], (56, 56), interpolation=cv2.INTER_CUBIC)
"""print(X_train[0].shape)
x = 0
for i in range(60000):
	X_train = cv2.resize(X_train[x], (56, 56), interpolation=cv2.INTER_CUBIC)
	x += 1
print(X_train.shape)"""

"""x = np.array([[1,5,3], [3,5,3]])
x = x.ravel()
x = x.reshape(2,3)
y = np.array([])
y = x[0]
y.reshape(())
print(y)
y = np.concatenate((y, x[1]))
print("h")
print(y)"""

#X_train1 = X_train.copy().ravel()
#y_train1 = y_train.copy().ravel()

#X_train2 = np.resize(X_train1, 56*56*60000)
#y_train2 = np.resize(y_train1, 56*56*60000)
#X_train1 = X_train.copy().ravel()

"""X_train = X_train.copy().ravel()
c = 0
k = 0
for i in range(len(X_train)):
	c += 1
	if c == 784:
		#tempe = X_train[:784]
		#X_train[k] = X_train[:784].reshape(28,28)
		X_train[k] = np.reshape(X_train[:784], (28,28))
		k += 1"""



"""for i in range(len(X_train)):
	#for j in range(len(X_train1[i])):
	X_train[i] = X_train[i].resize(56*56)
	k += 1
	if k == 10000:
		print(X_train[i].shape)"""

"""l = 0
for i in range(len(X_train)):
	l += 1
	#for j in range(len(X_train1[i])):
print(l)"""


#X_train2 = np.resize(X_train, 60000*56*56)
#y_train2 = np.resize(y_train, 60000*56*56)
"""print(X_train[0].shape)
#X_train = X_train2.resize(64*64*500).reshape(64, 64, 1)
#y_train = y_train2.resize(64*64*500).reshape(64, 64, 1)

X_train = X_train2.reshape(60000, 56, 56)
#y_train = y_train2.reshape(60000, 56, 56)

#print(X_train[0].shape)
#print(X_train.shape)
X_train[0].reshape((56,56))

plt.gray()
plt.imshow(X_train[0])
plt.show()"""

#rint(X_train[0].shape)

"""X_train = X_train.reshape(X_train.shape[0], 56, 56, 1)
print(X_train[0].shape)
X_test = X_test.reshape(X_test.shape[0], 56, 56, 1)"""
"""
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
print(X_train[0].shape)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#np.pad(X_train, ((0,0),(14,14),(14,14),(0,0)), "constant")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

number_of_classes = 10
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)"""