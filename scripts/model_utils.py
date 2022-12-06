import numpy as np
import tensorflow as tf
#from keras.models import load_model
import matplotlib.pyplot as plt
import os
import cv2
import sys
# Creating class for the script

class Model_3_Utils():
	def __init__(self, model: str):
		#model: name of .h5 model file
		self.model = tf.keras.models.load_model(model)
	
	def __str__(self):
		out = [layer.output_shape for layer in self.model.layers]
		result = '\n'.join(''.join(str(x)) for x in out)
		return "\nStructure of Alnet-3.0"

	def phase_one_preprocess(self, gray:np.ndarray)->np.ndarray:
		"""
		This is a helper function, dont call it directly.
		gray: grayscale image
		returns preprocessed image, ready for phase 2
		"""
		# dilate img
		dilated_gray = cv2.dilate(gray, np.ones((7,7), np.uint8))

		# median blur the dilated img to further suppress detail
		bg_gray = cv2.medianBlur(dilated_gray, 21)

		# calculate difference between gray(original img) and bg_gray.
		# identical pixels will be black(close to 0 difference), the digits details will be white(large difference).
		diff_gray = 255 - cv2.absdiff(gray, bg_gray)

		# normalize the image, so that we use the full dynamic range.
		norm_gray = diff_gray.copy()
		cv2.normalize(diff_gray, norm_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		# now the image is still gray in some areas, so we truncate that away and re-normalize the image.
		_, thr_gray = cv2.threshold(norm_gray, 253, 0, cv2.THRESH_TRUNC)
		cv2.normalize(thr_gray, thr_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

		# inverting color and rezising image to 112x112
		output = cv2.resize(255-thr_gray, (112, 112), interpolation=cv2.INTER_AREA)

		return output

	def format_to_input_layer(self, gray:np.ndarray)->np.ndarray:
		# expanding dimensions to fit the input layer

		# (WIDTH, HEIGHT, 1) -> (1, WIDTH, HEIGHT, 1)
		gray = np.expand_dims(gray, axis=0)

		#uint8 -> float32
		gray = gray.astype('float32')

		#normalize range to 0-1
		gray /= 255

		return gray

	def adaptive_preprocess(self, gray:np.ndarray)->np.ndarray:
		"""
		gray: grayscale image
		returns preprocessed image, ready for phase 2
		"""
		gray = self.phase_one_preprocess(gray)

		gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)

		return gray

	def absolute_preprocess(self, gray:np.ndarray)->np.ndarray:
		"""
		gray: grayscale image
		returns preprocessed image, ready for phase 2
		"""
		gray = self.phase_one_preprocess(gray)

		_, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		return gray

	def preprocess(self, gray:np.ndarray, method: str)->np.ndarray:
		"""
		gray: grayscale image
		method: "adaptive" or "absolute"
		returns preprocessed image, ready for input layer formating
		"""
		if method == "adaptive":
			return self.adaptive_preprocess(gray)
		elif method == "absolute":
			return self.absolute_preprocess(gray)
		else:
			raise ValueError("method must be 'adaptive' or 'absolute'")

	def predict(self, gray:np.ndarray, method: str)->np.ndarray:
		"""
		gray: grayscale image
		method: "adaptive" or "absolute"
		returns prediction
		"""
		gray = self.preprocess(gray, method)
		gray = self.format_to_input_layer(gray)
		return self.model.predict(gray)


if __name__ == "__main__":
	runner = Model_3_Utils("Alnet-gpu-3.0.h5")
	
	gray = cv2.imread("digits/2IMG_0623.jpg", 0)

	abs_thresh_img = runner.absolute_preprocess(gray)

	adapt_thresh_img = runner.adaptive_preprocess(gray)


