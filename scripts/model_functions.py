from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import cv2
import sys
# Creating class for the script
class Run():
	def __init__(self, model):
		self.model = model
		#self.model = load_model("ALnet-3.0.h5")
		"""
		print("\nStructure of Alnet-3.0")
		for layer in self.model.layers:
			print(layer.output_shape)
		"""
	
	def predict_chosen(self, path_to_file):
		"""
		
		"""
		filename = path_to_file
		# reading image to grayscale
		gray = cv2.imread(filename, 0)

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
		gray1 = cv2.resize(255-thr_gray, (112, 112), interpolation=cv2.INTER_AREA)

		# we have variables gray and gray2 to differentiate between two ways of thresholding.
		# Alnet-2.0 will make predictions on both of these, and combine the two to get our resulting classification.
		gray = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 3)
		(_, gray2) = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		
		# converting PIL image to numpy array
		img = image.img_to_array(gray)
		img2 = image.img_to_array(gray2)

		# reshaping to 1 channel (its only 1 image), 112x112 shape, and 1 color channel (grayscale)
		test_img = img.reshape((1,112,112,1))
		test_img2 = img2.reshape((1,112,112,1))

		# change datatype from uint8 to float32
		test_img = img.astype('float32')
		test_img2 = img2.astype('float32')

		# normalizing pixels in the decimal range 0 to 1
		test_img /= 255
		test_img2 /= 255

		# extend axis at index[0] to ready for prediction
		test_img = np.expand_dims(test_img, axis=0)
		test_img2 = np.expand_dims(test_img2, axis=0)

		# get network output (10 nodes)
		# prediction_list is prediction of gray
		# prediction2_list is prediction of gray2
		prediction_list = self.model.predict(test_img)
		prediction2_list = self.model.predict(test_img2)
		orig_img = cv2.imread(filename, 0)
		result = {"orig_img": orig_img, "prediction_list": prediction_list, "prediction2_list": prediction2_list, "adaptive_thresh_img": test_img, "thresh_img": test_img2}
		return result
		#return {"prediction_list": prediction_list, "prediction2_list": prediction2_list, "test_img": test_img, "test_img2": test_img2}

	def weighted_average(self, result):
		prediction_list = result["prediction_list"]
		prediction2_list = result["prediction2_list"]

		# x1 and x2 are indexes for elements (what digit is guessed) of highest value for images gray and gray2
		x1 = np.argmax(prediction_list)
		x2 = np.argmax(prediction2_list)

		# c is the highest value in prediction_list
		# v is the highest value in prediction2_list
		c = prediction_list[0][x1]
		v = prediction2_list[0][x2]

		# b is the value of prediction_list at index of prediction2_list highest value
		# n is the value of prediction2_list at index of predition_list highest value
		b = prediction_list[0][x2]
		n = prediction2_list[0][x1]

		# this helps us decide what prediction is most likely to be correct, by comparing the values below.
		k = (c+b)/2
		j = (v+n)/2
		if k > j:
			classname = x1

		if j > k:
			classname = x2

		result["classname"] = classname
		result["adaptive_thresh_acc"] = k
		result["thresh_acc"] = j
		return result
		#return {"classname": classname, "adaptive_thresh_acc": k, "thresh_acc": j, "adaptive_thresh_img": test_img, "thresh_img": test_img2}

	def predict_folder(self, pather, debug=False):
		"""
		pather is path to folder with images, 
		returns a dict of dicts: {"file1": {"classname": , "adaptive_thresh_acc": , "thresh_acc": , "adaptive_thresh_img": , "thresh_img": }, "file2": {...}}
		"""
		dir = os.listdir(pather)
		results = {}
		for filename in dir:
			if filename.endswith((".DS_Store")) or cv2.imread(f"{pather}/{filename}", 0) is None:
				pass
			else:
				results[filename] = self.weighted_average(
									self.predict_chosen(f"{pather}/{filename}")
									)
		return results

	def plot_comparison(self, weighted_data):
		# weighted data is same format as output of weighted_average()
		test_img = weighted_data["adaptive_thresh_img"]
		test_img2= weighted_data["thresh_img"]
		classname = weighted_data["classname"]
		k = weighted_data["adaptive_thresh_acc"]
		j = weighted_data["thresh_acc"]

		# reshaping again for showing image to user
		test_img = test_img.reshape((112,112))
		test_img2 = test_img2.reshape((112,112))

		# lines 181-195 displays to the user gray and gray2, values of k and j, and the estimated classname for the original image chosen
		fig = plt.figure(figsize=(8,8))
		columns = 3
		rows = 1
		plt.gray()
		fig.add_subplot(rows, columns, 1).set_title("adaptive_Threshold\n k = {}".format(k))
		plt.imshow(test_img)
		fig.add_subplot(rows, columns, 2).set_title("Threshold\n j = {}".format(j))
		plt.imshow(test_img2)
		fig.add_subplot(rows, columns, 3).set_title("Orig img")
		plt.imshow(weighted_data["orig_img"])
		if k > j:
			fig.suptitle("Image shows a {}\n k > j".format(classname))

		if j > k:
			fig.suptitle("Image shows a {}\n j > k".format(classname))
		fig.add_subplot(rows, columns, 3).set_title("Orig img")
		plt.imshow(weighted_data["orig_img"])
		plt.show()


if __name__ == "__main__":	
	location = os.path.abspath("")
	model = load_model(location+"/ALnet-3.0.h5")
	model = Run(model)

	x = model.predict_folder(f"{location}/digits")
	
	plt.gray()
	plt.imshow( x["2IMG_0341.JPG"]["orig_img"] )
	plt.show()