# cds-segmentation


## Segmentation algorithms

unetCarsTrain.py - training and testing of segmentation model in single python-script

BackgroundParser.py - data preparation from file with labeling and color and classes pallete

data.py - data generator

model.py - different segmentation models

segment_detect.py - segmentation and subsequent detection based on clustering and classification

cvat.py - functions for mask reading from xml cvat labeling

video_creator.py - python script for creation of video from two or one image sequences


## Conversion from keras to tensorflow and then to tensorrt

unetCarsTrain.py include conversion feature from keras to tensorflow .pb model

For conversion from tensorflow to tensorrt model use instruction: https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/

Som onput/output tensor names 

model with upsampling2D:

['input_1_1'] ['conv2d_13_1/Sigmoid']

model with Conv2DTranspose:

['input_1_1'] ['conv2d_11_1/Sigmoid']

model with Conv2DTranspose + Softmax and added layers:

['input_1_1'] ['conv2d_13_1/truediv']
