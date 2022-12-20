import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

print(X_train[0].shape)

#display 5 images
fig,ax = plt.subplots(2,5)
#print(ax.shape)
plt.gray()
for i in range(5):
    ax[0,i].imshow(X_train[i])

for i in range(5):
    img = X_train[i]
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ax[1,i].imshow(img)

plt.show()

