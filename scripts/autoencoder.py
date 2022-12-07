import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def data_prep():
    #previously the mnist classifier was trained on 112x112 images,
    #this to have easier preprocessing. Downscaling to 28x28 was unfeasible with thin pen strokes with noise.
    #the autoencoders job is to downscale this to the final 28x28 image, so we can have a mnist classifier which doesnt take years to train. (almost)

    #the autoencoder receives noisy 112x112 images, and outputs denoised 28x28 images.
    #the network design is not much thought behind, needs researching. Should it be similar to the digit classifier it feeds into?

    (X_traine, _), (X_teste, _) = tf.keras.datasets.mnist.load_data()

    X_train = np.empty((60000,112,112))
    X_test = np.empty((10000,112,112))

    X_traina = np.empty((60000,112,112))
    X_testa = np.empty((10000,112,112))

    kernel = np.ones((7,7),np.uint8)

    for i in range(60000):
        imgs = X_traine[i]
        enlargeder = cv2.resize(imgs, (112, 112), interpolation=cv2.INTER_AREA)
        erosion = cv2.erode(enlargeder,kernel,iterations = 1)
        X_train[i,:,:] = erosion
        X_traina[i] = enlargeder

    for i in range(10000):
        imgs = X_teste[i]
        enlargeder = cv2.resize(imgs, (112, 112), interpolation=cv2.INTER_AREA)
        erosion = cv2.erode(enlargeder,kernel,iterations = 1)
        X_test[i,:,:] = erosion
        X_testa[i] = enlargeder


    # reshaping for keras compatibility
    X_train = X_train.reshape(X_train.shape[0], 112, 112, 1)
    X_test = X_test.reshape(X_test.shape[0], 112, 112, 1)

    X_traina = X_traina.reshape(X_traina.shape[0], 112, 112, 1)
    X_testa = X_testa.reshape(X_testa.shape[0], 112, 112, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_traina = X_traina.astype('float32')
    X_testa = X_testa.astype('float32')

    # normalizing pixels
    X_train/=255
    X_test/=255

    X_traina/=255
    X_testa/=255
    
    noise_factor = 0.2
    X_train = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
    X_test = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 

    X_train = np.clip(X_train, 0., 1.)
    X_test = np.clip(X_test, 0., 1.)

    return X_train, X_traina, X_test, X_testa

@tf.function(jit_compile=True)
def tensorflow_setup():
    input_img = tf.keras.Input(shape=(112, 112, 1))

    x = tf.keras.layers.Conv2D(16, (20, 20), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(16, (10, 10), activation='relu', padding='same')(x)

    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)

    encoded = tf.keras.layers.BatchNormalization()(x)

    y = tf.keras.layers.Conv2DTranspose(32, (5, 5), activation='relu', padding='same')(encoded)
    y = tf.keras.layers.UpSampling2D((2, 2))(y)

    y = tf.keras.layers.Conv2DTranspose(16, (10, 10), activation='relu', padding='same')(y)
    y = tf.keras.layers.UpSampling2D((2, 2))(y)

    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2DTranspose(16, (20, 20), activation='relu', padding='same')(y)

    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(y)

    model = tf.keras.Model(input_img, decoded)
    return model

@tf.function(jit_compile=True)
def train_model():
    X_train_thin, X_train, X_test_thin, X_test = data_prep()
    with tf.device('/gpu:0'):
        autoencoder = tensorflow_setup()
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        history = autoencoder.fit(X_train_thin, X_train,
                        epochs=5,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(X_test_thin, X_test),
                        verbose=1)

    tf.keras.models.save_model(autoencoder, 'autoencoder.h5')

    return autoencoder, history, X_train_thin, X_train, X_test_thin, X_test




if __name__ == "__main__":
    
    autoencoder, history, X_train_thin, X_train, X_test_thin, X_test = train_model()
    decoded_imgs = autoencoder.predict(X_test_thin)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(X_test_thin[i].reshape(112, 112))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(112, 112))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()