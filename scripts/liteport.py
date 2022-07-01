import tensorflow as tf

import keras
x = keras.models.load_model("ALnet-3.0.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(x)
tfmodel = converter.convert()
open ("model.tflite", "wb").write(tfmodel)
#converter = tf.lite.TFLiteConverter.from_keras_model('ALnet-3.0.h5') 
#tfmodel = converter.convert()
#open ("model.tflite" , "wb") .write(tfmodel)

#x = keras.models.load_model("ALnet-3.0.h5")
#print(x)