# mnistcnn
handwritten digit classifier using with a convolutional neural network in keras.

ALnet-0.5 is in development for 1.0.
As of now its predictions are limited to:
  whiteboard images drawn in black (predict "digits/RGBsiffror" for demo).
  Digit needs to be "compact", otherwise the lines disappear during preprocessing,
  (predict "digits/" "siffrorsvåra", "whitetill", "svartsif", or "svartsif.jpg" for demo).
  images such as screenshots of jpg may have distinct borders as PNG, 
  and may be predicted incorrectly (predict "digits/pngsif" for demo).
