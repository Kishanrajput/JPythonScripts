# import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.layers import Lambda
import numpy as np

import numpy as np
import gzip
from struct import *

# Load a saved keras model
def loadKerasModel(path):
	# path --  path to saved model
	print("Loading Model path: ", path)
	m = tf.keras.models.load_model('MNISTmodel')
	print("model loaded")
	return m

# ------- Load Mninst Images ----------------
def loadImages(path):
	# path -- location of saved MNIST images

	rootdir = path
	setname="train-images-idx3-ubyte.gz"
	labelname="train-labels-idx1-ubyte.gz"
	images = gzip.open(rootdir+setname, "rb") 
	labels = gzip.open(rootdir+labelname, "rb") # Read the binary data 

	# We have to get big endian unsigned int. So we need ”>I”

	# Get metadata for images 
	images.read(4) # skip the magic number 
	number_of_images = images.read(4) 
	number_of_images = unpack(">I", number_of_images)[0] 
	rows = images.read(4) 
	rows = unpack(">I", rows)[0] 
	cols = images.read(4) 
	cols = unpack(">I", cols)[0]
	# Get metadata for labels 
	labels.read(4) # skip the magic number 
	N = labels.read(4) 
	N = unpack(">I", N)[0]
	if number_of_images != N: 
		raise Exception("number of labels did not match the number of images")
	# Get the data 
	x = np.zeros((N, rows, cols), dtype=np.ﬂoat32) # Initialize numpy array 
	y = np.zeros((N, 1), dtype=np.uint8) # Initialize numpy array 
	for i in range(1000): 
		if i % 1000 == 0: 
			print("i: %i" % i)
		tmp_label = labels.read(1) 
		if tmp_label != b'':
			y[i] = unpack(">B", tmp_label)[0]
		for row in range(rows): 
			for col in range(cols): 
				tmp_pixel = images.read(1) # Just a single byte 
				tmp_pixel = unpack(">B", tmp_pixel)[0] 
				x[i][row][col] = tmp_pixel 
	return (x, y)




