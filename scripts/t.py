import tensorflow as tf
import os
import tensorflow_datasets as tfds
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

train_ds = tfds.load('mnist', split='train', shuffle_files=True)
val_ds = tfds.load('mnist', split='test', shuffle_files=True)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

#minskar io bottleneck
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

tf.config.run_functions_eagerly(True)

for example in train_ds.take(1):
    image, label = example["image"], example["label"]
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())