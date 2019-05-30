# mnistcnn
handwritten digit classifier using with a convolutional neural network in keras.

ALnet-0.5 is in development for 2.0.
As of now its predictions are limited to:
  whiteboard images drawn in black (predict "digits/RGBsiffror" for demo).
  Digit needs to be "compact", otherwise the lines disappear during preprocessing,
  (predict "digits/" "siffrorsvåra", "whitetill", "svartsif", or "svartsif.jpg" for demo).
  images such as screenshots of jpg may have distinct borders as PNG, 
  and may be predicted incorrectly (predict "digits/pngsif" for demo).
  
  
UPDATE! Alnet-2.0 is soon to be done.
As of now the the mnist images have been upscaled to 112x112 before training so that input images now are JPGs preprocessed to 112x112 instead of 28x28 which combines with strided convolutions to solve the issue of thin lines disappearing during preprocessing.

  To install dependencies, download the project and in terminal run pip3 install -r path to requirements.txt, for example:  
  user $ pip3 install -r /Users/username/Downloads/mnistcnn-master/requirements.txt

IMPORTANT! Currently issues with windows compatibility.

Work In Progress...
