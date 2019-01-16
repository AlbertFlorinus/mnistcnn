import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
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

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
#print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255


number_of_classes = 10

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)


# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

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


"""gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=1, 
                    validation_data=test_generator, validation_steps=10000//64, shuffle=True)
model.save("model-cnn2.h5")"""

model.load_weights("model-cnn2.h5")
from scipy.ndimage import rotate
#img = image.load_img(path="nians.JPG",color_mode="grayscale",target_size=(28,28))
bp = input("Vilken bild? ")
img = image.load_img(path=bp,color_mode="grayscale",target_size=(28,28))
if bp.endswith("JPG"):
	img = rotate(img, 270)
img = image.img_to_array(img)
img = 255-img
#np.rot90(img,axes=(-2,-1))
img[np.where((img==[255.]).all(axis=2))] = [40.]
img[np.where((img < [100.]).all(axis=2))] = [20.]
#img[3] = [40.]
#img[24] = [40.]
#img[0:4] = img[4]
#img[23:28] = img[22]
print(img)
print(img.shape)
test_img = img.reshape((1,28,28,1))
test_img = img.astype('float32')
test_img/=255
print(test_img.shape)
test_img = np.expand_dims(test_img, axis=0)
print(test_img.shape)
img_class = model.predict(test_img, batch_size=1, verbose=1)
prediction = img_class
print(prediction)
classname = np.argmax(prediction)
print("Class: ",classname)
img = img.reshape((28,28))
plt.gray()
plt.imshow(img)
plt.title(classname)
plt.show()

"""from keras import models
model.load_weights("model-cnn.h5")
img = image.load_img(path="sju.JPG",color_mode="grayscale",target_size=(28,28))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

#plt.imshow(img_tensor)
#plt.show()
print(img_tensor.shape)
layer_outputs = [layer.output for layer in model.layers[:12]] # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activationactivations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[8]
print(first_layer_activation.shape)
print(first_layer_activation)
layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            #channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.gray()
    plt.imshow(display_grid, aspect='auto', cmap='binary')
    plt.show()"""