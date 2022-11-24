import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# gathering the mnist-dataset, train is used for training,
# test is used to predict images previously unseen,
# we do this to ensure overfitting has not occurred,
# y are labels to X which are the images

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

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

def keras_setup():        
    model = tf.keras.Sequential()

    # each convolutional/conv layer distorts the input arrays
    # each model.add creates a new layer in the network
    # adding a spatial convolutional/conv layer and 
    # declaring input shape required for the img array
    # input shape value only needed for first conv
    # 32 for the amount of filters and thus also feature maps outputed,
    # (3,3) for filter size, default stride is 1

    #model.add(Conv2D(16, (20,20), strides = 2, activation="relu", input_shape=(112,112,1), padding="same"))
    #model.add(Conv2D(16, (10,10), strides = 2, activation="relu", padding="same"))
    model.add(Conv2D(16, (3,3), activation="relu", padding="same", input_shape=(28,28,1)))
    model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
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

    return model

def tensorflow_setup():
    input_img = tf.keras.Input(shape=(28,28,1))
    x = Conv2D(16, (3,3), activation="relu", padding="same")(input_img)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (5,5), strides=2, padding="same",activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv2D(64, (3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (5,5), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(number_of_classes, activation="softmax")(x)
    model = tf.keras.Model(input_img, output)
    return model

if __name__ == "__main__":

    # Number of images to iterate simultaneously before each weight update
    batchsize = 32

    X_val = X_test[9000:]
    Y_val = Y_test[9000:]

    X_test = X_test[:9000]
    Y_test = Y_test[:9000]

    # Augmentning training data for better generalisation,
    # and prevent overfitting
    gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.15)
    train_generator = gen.flow(X_train, Y_train, batch_size=batchsize)
    testing_generator = gen.flow(X_test, Y_test, batch_size=batchsize)

    # Reducing learning rate to 95% of the last epoch,
    # speeding up convergence by keeping weight updates smaller as the model,
    # approaches convergence.
    annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    # Log file for tracking information about the learning process and its metrics
    csv_logger = tf.keras.callbacks.CSVLogger("training_test_4.0.log", append=True, separator=";")


    model = tensorflow_setup()

    with tf.device('/gpu:0'):
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # starting training
        history = model.fit(train_generator,steps_per_epoch=X_train.shape[0]//batchsize, epochs=10, 
                        validation_data=testing_generator, callbacks=[annealer, csv_logger], verbose=1)

        model.save("ALnet-4.0.h5")
    
    score = model.evaluate(x=X_val, y=Y_val, verbose=1)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
