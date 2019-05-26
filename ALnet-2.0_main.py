import os
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model
import model_functions


# alnet-0.5 is in development for 1.0.
# As of now its predictions are limited to:
# whiteboard images drawn in black (predict "digits/RGBsiffror" for demo).
# digit needs to be "compact", otherwise the lines,
# disappear during preprocessing,
# (predict "digits/" "siffrorsvåra", "whitetill", "svartsif", or "svartsif.jpg" for demo).
# images such as screenshots of jpg may have distinct,
# borders as PNG, and may be predicted incorrectly (predict "digits/pngsif" for demo).
# Also some sensitivity to noise

run = model_functions.Run()
# To predict a single image
# double click to select file
# Currently a bug where you need to select the image twice

#run.predict_chosen()
#run.predict_folder("digits")

# To predict a folder of images,
# As of alnet-0.5, this is in development,
# currently the image files needs to start with the correct number,
# in order to get results to a list, 
# the model however doesnt use the image filename for its prediction!
# "pngsif" folder isnt made for this function!
# the path to the folder is specified in place of path,
# example folder of "siffror" is predicted as: 
# run.predict_folder("digits/RGBsiffror")

#run.predict_folder("digits2")

