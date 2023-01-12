import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

def build_dataset(dirPath: str, IMG_SIZE = (112,112), BATCH_SIZE = 1):
    def gray_transform(inputs):
        x = tf.image.rgb_to_grayscale(inputs)
        x = tf.squeeze(x)
        x = tf.repeat(x[:,:,:,np.newaxis], 3, axis=3)
        return x

    def normalize(inputs):
        inputs = tf.cast(inputs, tf.float32)
        inputs = inputs / 255.0
        return inputs

    test_ds = tf.keras.utils.image_dataset_from_directory(dirPath,
                                                            batch_size = BATCH_SIZE, image_size = IMG_SIZE,label_mode = "categorical",
                                                      labels="inferred", color_mode = "grayscale").map(lambda x,y: (normalize(x),y))

    return test_ds


dirPathx = "data/digits_112_absolute"
dirPathy = "data/digits_112_adaptive"
dirPathz = "data/digits_tensordata"
test_ds = build_dataset(dirPathx)
test_ds2 = build_dataset(dirPathy)
test_ds3 = build_dataset(dirPathz)
model = tf.keras.models.load_model("ALnet-3.0.h5")

#evaluate the model
loss, acc = model.evaluate(test_ds)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
loss, acc = model.evaluate(test_ds2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
loss, acc = model.evaluate(test_ds3)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))