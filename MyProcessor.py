
import jana
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.layers import Lambda
import numpy as np
import time
import sys
#from DHit import *
#from DCluster import *

# ---- Import My Python Script -------------------
import PredictMnist as pm
# ------------------------------------------------

# JEventProcessor class defined in janapy module
class MyProcessor(jana.JEventProcessor):
	def __init__(self):
		super().__init__(self)

	def Init( self ):
		print('Python Init called')
		self.data, self.labels = pm.loadImages('Data/')
		self.model = pm.loadKerasModel('MNISTmodel')
		self.counter = 0
		# self.predictions = []
		self.start_time = time.time()
		print("Length of data: ", len(self.data))
		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# session = tf.Session(config=config)
		# K.set_session(session)
		# time.sleep(10)


	# event is a JEvent object defined in janapy module
	def Process( self ):
		
		print('Python Process called')



		# model = pm.loadKerasModel('MNISTmodel copy'+str(self.counter))
		# model = pm.loadKerasModel('MNISTmodel')
		# with tf.session() as sess:
		# 	K.set_session()
		# time.sleep(20)
		# model = tf.keras.models.load_model('MNISTmodel')
		# predictions = self.model.predict(data[self.counter:self.counter+20])
		step = 10000
		if self.counter > 50000:
			print("Processing time: ", time.time()-self.start_time)
			sys.exit()
		l = self.labels[self.counter:self.counter+step]
		# d = self.data[self.counter:self.counter+step]
		predictions = self.model.predict(self.data[self.counter:self.counter+step])
		# # # self.predictions.append(predictions)
		for i in range(len(predictions)):
			print(np.argmax(predictions[i]), l[i])
		# # # print(predictions, l)
		self.counter += step


		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			# Restrict TensorFlow to only use the first GPU
			try:
				tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
			except:
				print("Error")
		else:
			print("No GPU")
		
		# # 	print(l[i])
		
		print(self.counter)

	def Finish( self ):
		print('Python Finish called')

#		hits = event.Get( DHit )
#		for h in hits:
#			print('hit:  a=%d  b=%f  type=%s' % (h.a, h.b , type(h)))

#-----------------------------------------------------------

jana.SetParameterValue( 'JANA:DEBUG_PLUGIN_LOADING', '1')
jana.SetParameterValue( 'NTHREADS', '4')
jana.SetParameterValue( 'NEVENTS', '100')

# The janapy module itself serves as the JApplication facade
jana.AddProcessor( MyProcessor() )

jana.AddPlugin('jtest')
jana.Run()

print('PYTHON DONE.')

