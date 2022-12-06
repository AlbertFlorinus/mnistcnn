import numpy as np
import scipy
import tensorflow as tf
import cv2


(X_traine, Y_train), (X_teste, Y_test) = tf.keras.datasets.mnist.load_data()


#upscaling the mnist dataset from 28x28 pixel images to 112x112
#this is a bandaid fix but helps with preprocessing real images outside mnist
X_train = np.empty((60000,112,112))
X_test = np.empty((10000,112,112))

for i in range(60000):
    imgs = X_traine[i]
    enlargeder = cv2.resize(imgs, (112, 112), interpolation=cv2.INTER_AREA)
    X_train[i,:,:] = enlargeder

for i in range(10000):
    imgs = X_teste[i]
    enlargeder = cv2.resize(imgs, (112, 112), interpolation=cv2.INTER_AREA)
    X_test[i,:,:] = enlargeder
    
# reshaping for keras compatibility
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

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
Y_train = tf.keras.utils.to_categorical(Y_train, number_of_classes)
Y_test = tf.keras.utils.to_categorical(Y_test, number_of_classes)

def tensorflow_setup():
    input_img = tf.keras.Input(shape=(112,112,1))

    x = tf.keras.layers.Conv2D(16, (20,20), strides = 2, activation="relu", padding="same")(input_img)
    x = tf.keras.layers.Conv2D(16, (10,10), strides = 2, activation="relu", padding="same")(x)

    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # adding conv layer with 5x5 filter and a stride of 2 instead of max pooling,
    # downsampling image but retaining import data for classification.
    x = tf.keras.layers.Conv2D(32, (5,5), strides=2, padding="same",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # using dropout with a rate 0f 0.4, this randomly "drops",
    # 40% of the nodes to a output value of 0 each iteration, which helps prevent overfitting
    x = tf.keras.layers.Dropout(0.4)(x)

    # raise amount from 32 to 64
    x = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, (5,5), strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # flattening the input to a 1d array,
    # flattening the pixel data of 64 4x4 arrays
    # to a 1d array containing 1024 pixel values,
    # not 1024 pixels of the original image but of the,
    # outputs from the convolutional neural network
    # this ends the spatial/convolutional part of the network
    x = tf.keras.layers.Flatten()(x)

    # adding a fully connected layer of 128 neurons
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Final layer of 10 neurons with a softmax activation,
    output = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)
    model = tf.keras.Model(input_img, output)
    return model

if __name__ == "__main__":
    # Number of images to iterate simultaneously before each weight update
    batchsize = 64

    X_val = X_test[:9000]
    Y_val = Y_test[:9000]

    X_test = X_test[9000:]
    Y_test = Y_test[9000:]

    # Augmentning training data for better generalisation,
    # and prevent overfitting
    gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.15)
    train_generator = gen.flow(X_train, Y_train, batch_size=batchsize)
    val_generator = gen.flow(X_val, Y_val, batch_size=batchsize)

    # Reducing learning rate to 95% of the last epoch,
    # speeding up convergence by keeping weight updates smaller as the model,
    # approaches convergence.
    annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    model = tensorflow_setup()

    if tf.config.list_physical_devices('GPU'):
        #make sure you have cuda installed and are using tensorflow-gpu
        with tf.device('/gpu:0'):
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            history = model.fit(train_generator,steps_per_epoch=X_train.shape[0]//batchsize, epochs=10, 
                            validation_data=val_generator, callbacks=[annealer], verbose=1)

        model.save("ALnet-gpu-3.0.h5")

    else:
        print("No GPU found, using CPU instead")    
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(train_generator,steps_per_epoch=X_train.shape[0]//batchsize, epochs=10, 
                validation_data=val_generator, callbacks=[annealer], verbose=1)

        model.save("ALnet-cpu-3.0.h5")

    score = model.evaluate(x=X_test, y=Y_test, verbose=1)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
