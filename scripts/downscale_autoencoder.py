import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def data_prep():
    X_traine: np.ndarray
    X_teste: np.ndarray
    (X_traine, _), (X_teste, _) = tf.keras.datasets.mnist.load_data()
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
    X_train = X_train.reshape(X_train.shape[0], 112, 112, 1)
    X_test = X_test.reshape(X_test.shape[0], 112, 112, 1)

    X_traine = X_traine.reshape(X_traine.shape[0], 28, 28, 1)
    X_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

    # changing datatype from uint8 to float32,
    # this is to allow for normalizising the pixel value,
    # between 0 and 1
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_traine = X_traine.astype('float32')
    X_teste = X_teste.astype('float32')

    # normalizing pixels
    X_train/=255
    X_test/=255

    X_traine/=255
    X_teste/=255

    #Adding noise
    noise_factor = 0.5
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 

    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)

    # one-hot-encoding the outputs, or in simpler terms,
    # storing the outputs as a vector of 1x10
    number_of_classes = 10
    #Y_train = tf.keras.utils.to_categorical(Y_train, number_of_classes)
    #Y_test = tf.keras.utils.to_categorical(Y_test, number_of_classes)

    #return X_train, Y_train, X_test, Y_test
    return X_train_noisy, X_traine, X_test_noisy, X_teste







input_img = tf.keras.Input(shape=(112, 112, 1))

x = tf.keras.layers.Conv2D(16, (20, 20), strides = 2, activation='relu', padding='same')(input_img)
x = tf.keras.layers.Conv2D(16, (10,10), strides = 2, activation="relu", padding="valid")(x)

x = tf.keras.layers.Conv2D(32, (5,5), strides = 2, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(32, (5,5), strides = 1, padding = "same", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Conv2D(64, (3,3), strides = 1, padding = "same", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)

x =  tf.keras.layers.Conv2D(64, (3,3), strides = 1, padding = "valid", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
#x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
#x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

xup = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(xup)
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="valid")(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


#autoencoder = tf.keras.Model(input_img, decoded)
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#print(autoencoder.summary())
if __name__ == '__main__':
    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    #X_train, Y_train, X_test, Y_test = data_prep()
    X_train_noisy, X_traine, X_test_noisy, X_teste = data_prep() #noisy is upscaled images with noise, e is the original mnist images
    print(f"X_train_noisy shape: {X_train_noisy.shape}")
    print(f"X_traine shape: {X_traine.shape}")
    print(f"X_test_noisy shape: {X_test_noisy.shape}")
    print(f"X_teste shape: {X_teste.shape}")
    history = autoencoder.fit(X_train_noisy, X_traine,
                    epochs=3,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_test_noisy, X_teste),
                    verbose=1)

    tf.keras.models.save_model(autoencoder, 'downscaling_autoencoder.h5')
    #with tf.device("/CPU:0"):
    decoded_imgs = autoencoder.predict(X_test_noisy)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(X_test_noisy[i].reshape(112, 112))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()