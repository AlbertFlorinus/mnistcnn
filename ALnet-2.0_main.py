import os
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model
import model_functions

# Alnet-2.0 is limited to images with bright (preferably white) backgrounds with dark digits (preferably black or dark-gray)
# bad lightning seems to work fine
# Images such as screenshots of jpg may have distinct,
# borders in .PNG format, and may be predicted incorrectly.
# Also some sensitivity to distinct lines in the background.

run = model_functions.Run()

# To predict a single image
# Currently a bug where you need to select the image twice
run.predict_chosen()

# remove the "#" from line 21 to predict all images in the digits folder
#run.predict_folder("digits")

