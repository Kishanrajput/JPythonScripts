import jana
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.layers import Lambda
import numpy as np
import time
#from DHit import *
#from DCluster import *

# ---- Import My Python Script -------------------
import PredictMnist as pm
# ------------------------------------------------


data, labels = pm.loadImages('Data/')
model = pm.loadKerasModel('MNISTmodel')
counter = 0
pred = []
start_time = time.time()


for i in range(5):
	step = 10000
	predictions = model.predict(data)
	l = labels[counter:counter+step]
	pred.append(predictions)
	counter += step
# for i in range(len(predictions)):
# 	print(np.argmax(predictions[i]), l[i])
print("Processing time: ", time.time()-start_time)