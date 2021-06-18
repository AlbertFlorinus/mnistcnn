def preprocessor(img):
    # this resize the image to square 112x112 shape with interpolating to center the digit
    gray = img
    dilated_gray = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg_gray = cv2.medianBlur(dilated_gray, 21)
    diff_gray = 255-cv2.absdiff(gray, bg_gray)
    norm_gray = diff_gray.copy()
    norm_img = np.zeros((320,240))
    norm_gray = cv2.normalize(diff_gray, norm_img, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    test_img = diff_gray - norm_gray  
    test_img2 = cv2.dilate(test_img, np.ones((5,5)))

    ret2,gray3 = cv2.threshold(test_img2,0,255,cv2.THRESH_OTSU)
    gray3 = cv2.resize(gray3, (28,28), interpolation = cv2.INTER_AREA)
    return gray3, gray3.copy()

def into_alnet(img):
    img = img.reshape((1,28,28,1))
    img = img.astype('float32')
    img /= 255
    return img

import os
cnet_in = []
usr_out = []
usr_orig = []
count = 0
for filename in os.listdir("/content/digits/digits"):  
    if filename.endswith((".DS_Store")):
        pass
    elif count < 1:
        count += 1
    elif count < 120:

        img = cv2.imread("/content/digits/digits/" + filename, 0)
        prep, usr_img = preprocessor(img)
        prep = into_alnet(prep)
        cnet_in.append(prep)
        usr_orig.append(img)
        usr_out.append(usr_img)

        count += 1
    
import matplotlib.pyplot as plt

count = 0

for i in cnet_in:
  fig = plt.figure(figsize=(8,8))
  plt.gray()
  prediction = model.predict(i)
  #print(prediction)
  x1 = np.argmax(prediction)

  fig.add_subplot(1,2,1).set_title(x1)
  plt.imshow(usr_out[count])

  fig.add_subplot(1,2,2)
  plt.imshow(usr_orig[count])
  #print(x1)
  count += 1