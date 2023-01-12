import os
import cv2
import sys
import tensorflow as tf

dirPath = "digits_tensordata/"

def build_dataset(dirPath: str, IMG_SIZE = (240,320), BATCH_SIZE = 1):
    test_ds = tf.keras.utils.image_dataset_from_directory(dirPath,
                                                            batch_size = BATCH_SIZE, image_size = IMG_SIZE,label_mode = "categorical",
                                                      labels="inferred")

    mapper = test_ds.class_names


    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))
    print("Number of samples: %d", len(test_ds)*BATCH_SIZE)
    #minskar io bottleneck
    AUTOTUNE = tf.data.AUTOTUNE

    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    return test_ds, mapper

test_ds, mapper = build_dataset(dirPath)
print(mapper)
