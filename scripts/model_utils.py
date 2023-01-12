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
	"""
	runner = Model_3_Utils("Alnet-gpu-3.0.h5")

	#filling absolute preprocessed directory

	for dir in os.listdir("data/digits_112_absolute"):
		if dir == ".DS_Store":
			continue
		for file in os.listdir("data/digits_tensordata/"+dir):
			gray = cv2.imread("data/digits_tensordata/"+dir+"/"+file, cv2.IMREAD_GRAYSCALE)
			gray = runner.absolute_preprocess(gray)
			cv2.imwrite("data/digits_112_absolute/"+dir+"/"+file, gray)
	
	"""

	"""
	#filling absolute preprocessed directory
	runner = Model_3_Utils("Alnet-gpu-3.0.h5")
	for dir in os.listdir("data/digits_112_adaptive"):
		if dir == ".DS_Store":
			continue
		for file in os.listdir("data/digits_tensordata/"+dir):
			gray = cv2.imread("data/digits_tensordata/"+dir+"/"+file, cv2.IMREAD_GRAYSCALE)
			gray = runner.adaptive_preprocess(gray)
			cv2.imwrite("data/digits_112_adaptive/"+dir+"/"+file, gray)
	"""

if __name__ == "__main__":
	correct = 0
	wrong = 0
	runner = Model_3_Utils("Alnet-gpu-3.0.h5")
	autoencoder_eroder = tf.keras.models.load_model("downscaling_autoencoder.h5")
	autoencoder_scaler = tf.keras.models.load_model("downscaling_autoencoder_2.h5")
	

	model = tf.keras.models.load_model("ALnet-4.0.h5")
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	autoencoder_eroder.compile(optimizer='adam', loss='binary_crossentropy')
	autoencoder_scaler.compile(optimizer='adam', loss='binary_crossentropy')

	for filename in os.listdir("digits"):
		if filename.endswith(".DS_Store"):
			continue
		gray = cv2.imread("digits/" + filename, 0)
		img = runner.adaptive_preprocess(gray)
		#img = runner.absolute_preprocess(gray)
		img = img.astype("float32")/255
		img = np.expand_dims(img, axis=0)
		#img = autoencoder_scaler.predict(img, verbose=0)[0]
		img = autoencoder_eroder.predict(img, verbose = 0)[0]
		img = np.expand_dims(img, axis=0)
		res = model.predict(img, verbose=0)[0]
		prediction = np.argmax(res)
		if int(prediction) == int(filename[0]):
			correct += 1
		else:
			wrong += 1
	print(correct/(wrong+correct))

if __name__ != "__main__":
	runner = Model_3_Utils("Alnet-gpu-3.0.h5")
	autoencoder_eroder = tf.keras.models.load_model("downscaling_autoencoder.h5")
	autoencoder_scaler = tf.keras.models.load_model("downscaling_autoencoder_2.h5")
	
	autoencoder_eroder.compile(optimizer='adam', loss='binary_crossentropy')
	autoencoder_scaler.compile(optimizer='adam', loss='binary_crossentropy')

	#plot 3x10 images
	fig, ax = plt.subplots(5, 10, figsize=(12, 4))
	#fig.subplots_adjust(hspace = .00005, wspace=.005)
	#plt.tight_layout()
	count = 0
	for filename in os.listdir("digits")[80:]:
		if filename.endswith(".DS_Store"):
			continue
		elif count < 10:
			gray = cv2.imread("digits/" + filename, 0)
			abs_thresh_img = runner.absolute_preprocess(gray)
			adapt_thresh_img = runner.adaptive_preprocess(gray)
			
			abs_thresh_coded = abs_thresh_img.astype('float32') / 255
			abs_thresh_coded = np.expand_dims(abs_thresh_coded, axis=0)
			abs_thresh_coded_eroder = autoencoder_eroder.predict(abs_thresh_coded)[0]
			abs_thresh_coded_scaler = autoencoder_scaler.predict(abs_thresh_coded)[0]

			adapt_thresh_coded = adapt_thresh_img.astype('float32') / 255
			adapt_thresh_coded = np.expand_dims(adapt_thresh_coded, axis=0)
			adapt_thresh_coded_eroder = autoencoder_eroder.predict(adapt_thresh_coded)[0]
			adapt_thresh_coded_scaler = autoencoder_scaler.predict(adapt_thresh_coded)[0]	

			ax[0, count].imshow(abs_thresh_coded_eroder, cmap="gray")
			ax[1, count].imshow(abs_thresh_coded_scaler, cmap="gray")

			ax[2, count].imshow(adapt_thresh_coded_eroder, cmap="gray")
			ax[3, count].imshow(adapt_thresh_coded_scaler, cmap="gray")
			ax[4, count].imshow(gray, cmap="gray")

			count += 1

	#hide x and y ticks
	for i in range(5):
		for j in range(10):
			ax[i, j].set_xticks([])
			ax[i, j].set_yticks([])

	#title for each row
	ax[0, 0].set_title("abs+eroder", fontsize=15)
	ax[1, 0].set_title("Abs+scaler", fontsize=15)
	ax[2, 0].set_title("Adapt+eroder", fontsize=15)
	ax[3, 0].set_title("Adapt+scaler", fontsize=15)
	ax[4, 0].set_title("Original Image", fontsize=15)
	plt.show()

if __name__ != "__main__":
	runner = Model_3_Utils("Alnet-gpu-3.0.h5")
	autoencoder = tf.keras.models.load_model("downscaling_autoencoder_2.h5")
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
	
	#plot 3x10 images
	fig, ax = plt.subplots(5, 10, figsize=(12, 4))
	#fig.subplots_adjust(hspace = .00005, wspace=.005)
	#plt.tight_layout()
	count = 0
	for filename in os.listdir("digits")[50:]:
		if filename.endswith(".DS_Store"):
			continue
		elif count < 10:
			gray = cv2.imread("digits/" + filename, 0)
			abs_thresh_img = runner.absolute_preprocess(gray)
			adapt_thresh_img = runner.adaptive_preprocess(gray)
			
			abs_thresh_coded = abs_thresh_img.astype('float32') / 255
			abs_thresh_coded = np.expand_dims(abs_thresh_coded, axis=0)
			abs_thresh_coded = autoencoder.predict(abs_thresh_coded)[0]

			adapt_thresh_coded = adapt_thresh_img.astype('float32') / 255
			adapt_thresh_coded = np.expand_dims(adapt_thresh_coded, axis=0)
			adapt_thresh_coded = autoencoder.predict(adapt_thresh_coded)[0]

			ax[0, count].imshow(abs_thresh_img, cmap="gray")
			ax[1, count].imshow(abs_thresh_coded, cmap="gray")

			ax[2, count].imshow(adapt_thresh_img, cmap="gray")
			ax[3, count].imshow(adapt_thresh_coded, cmap="gray")
			ax[4, count].imshow(gray, cmap="gray")

			count += 1

	#hide x and y ticks
	for i in range(5):
		for j in range(10):
			ax[i, j].set_xticks([])
			ax[i, j].set_yticks([])

	#title for each row
	ax[0, 0].set_title("Absolute Thresholding", fontsize=15)
	ax[1, 0].set_title("Abs Thresholding + Autoencoder", fontsize=15)
	ax[2, 0].set_title("Adaptive Thresholding", fontsize=15)
	ax[3, 0].set_title("Adapt Thresholding + Autoencoder", fontsize=15)
	ax[4, 0].set_title("Original Image", fontsize=15)
	plt.show()


if __name__ != "__main__":
	runner = Model_3_Utils("Alnet-gpu-3.0.h5")
	
	gray = cv2.imread("digits/6IMG_0664.jpg", 0)

	abs_thresh_img = runner.absolute_preprocess(gray)

	adapt_thresh_img = runner.adaptive_preprocess(gray)

	abs_inp = runner.format_to_input_layer(abs_thresh_img)

	adapt_inp = runner.format_to_input_layer(adapt_thresh_img)

	autoencoder = tf.keras.models.load_model("autoencoder.h5")

	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

	reduce_imgabs = autoencoder.predict(abs_inp)[0]

	reduce_imgabs = cv2.resize(reduce_imgabs, (28, 28), interpolation=cv2.INTER_AREA)

	reduce_imgadapt = autoencoder.predict(adapt_inp)[0]

	reduce_imgadapt = cv2.resize(reduce_imgadapt, (28, 28), interpolation=cv2.INTER_AREA)

	classifier = tf.keras.models.load_model("ALnet-4.0.h5")

	#classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	#print(reduce_imgabs.shape)
	#print("class is : ",classifier.predict(runner.format_to_input_layer(reduce_imgabs)))
	print(classifier.predict(reduce_imgabs.reshape(1, 28, 28, 1)))
	plt.subplot(141)
	plt.imshow(abs_thresh_img, cmap='gray')
	plt.subplot(142)
	plt.imshow(reduce_imgabs, cmap='gray')
	plt.subplot(143)
	plt.imshow(adapt_thresh_img, cmap='gray')
	plt.subplot(144)
	plt.imshow(reduce_imgadapt, cmap='gray')
	plt.show()
	#autoencoder.fit(abs_inp, abs_inp, epochs=10, batch_size=32)

	#adapt_thresh_img = runner.adaptive_preprocess(gray)


