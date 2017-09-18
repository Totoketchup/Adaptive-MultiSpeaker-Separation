# My Model 
from utils.ops import ops
from utils.ops.ops import Residual_Net, Conv1D, Reshape, Dense, unpool
from tensorflow.contrib.tensorboard.plugins import projector
# from utils.postprocessing.reconstruction import 
# import matplotlib.pyplot as plt

import os
import config
import tensorflow as tf
import time 
import soundfile as sf
import numpy as np

#############################################
#     Adaptive Front and Back End Model     #
#############################################

class Adapt:

	def __init__(self):

		self.N = 256
		self.max_pool_value = self.N

		self.graph = tf.Graph()

		with self.graph.as_default():
			# Batch of raw audio - Input data
			# shape = [ batch size , samples ]
			self.X_raw = tf.placeholder("float", [None, None])

			self.front
			self.separator
			self.back

			self.cost
			self.optimize

			self.saver = tf.train.Saver()
			self.merged = tf.summary.merge_all()

			self.train_writer = tf.summary.FileWriter('log/adaptive', self.graph)

		# Create a session for this model based on the constructed graph
		self.sess = tf.Session(graph = self.graph)


	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())

	##
	## Front End creating STFT like data + P Matrix
	##
	@ops.scope
	def front(self): 
		# Front-End Adapt

		# X_raw : [ B , T , 1]
		X_in = tf.expand_dims(self.X_raw, 2)
		self.input_shape = tf.shape(X_in)
        
        # 1 Dimensional convolution along T axis with a window length = 1024
        # And N filters
		self.WC1 = tf.Variable(tf.random_normal([1024, 1, self.N], stddev=0.35))
		X = tf.nn.conv1d(X_in, self.WC1, stride=1, padding="SAME") 

		# X : [ B , T , N]
		X_abs = tf.abs(X)
        
        # Smoothing the 'STFT' like created by the previous OP
		smoothing_filter = tf.Variable(tf.random_normal([1,4,1,1], stddev=0.35))
		M = tf.nn.conv2d(tf.expand_dims(X_abs,3), smoothing_filter, strides=[1, 1, 1, 1], padding="SAME")
     	
		M = tf.nn.softplus(M)

		# Matrix for the reconstruction process P = M / X => X * P = M
		self.P = M / tf.expand_dims(X,3)
        
        # Max Pooling
		y, self.argmax = tf.nn.max_pool_with_argmax(M, (1, max_pool_value, 1, 1), strides=[1, max_pool_value, 1, 1], padding="SAME")
        
		return tf.squeeze(y, [3])

	@ops.scope
	def separator(self):
		## ##
		## Signal separator network for testing.
		## ##
		shape = tf.shape(self.front)
		layers = [
			Dense(self.N, 500, tf.nn.softplus),
			Dense(500, 500, tf.nn.softplus),
			Dense(500, self.N, tf.nn.softplus)
		]

		def f_props(layers, x):
			for i, layer in enumerate(layers):
				x = layer.f_prop(x)
			return x

		separator_out = f_props(layers, tf.reshape(self.front, [-1,self.N]))
		return tf.expand_dims(tf.reshape(separator_out, shape),3)

	@ops.scope
	def back(self):
		# Back-End 

		# Unpooling (the previous max pooling)
		unpooled = unpool(self.separator, self.argmax, ksize=[1, self.max_pool_value, 1, 1], scope='unpool')
		
		out = unpooled * self.P

		out = tf.reshape(out, [1, self.input_shape[0], self.N, 1])
		return tf.nn.conv2d_transpose(tf.transpose(out, [0,1,3,2]), filter=tf.expand_dims(self.WC1,0), output_shape=[1, self.input_shape[0], 1, 1], strides=[1,1,1,1])

	@ops.scope
	def cost(self):
		# Definition of cost for Adapt model
		cost = tf.nn.l2_loss(self.X_raw - self.back) + 0.01*tf.norm(self.X_raw)
		tf.summary.scalar('training cost', cost)
		return cost

	@ops.scope
	def optimize(self):
		return tf.train.AdamOptimizer().minimize(self.cost)

	def save(self, step):
		self.saver.save(self.sess, os.path.join('log/adaptive/', "adaptive_model.ckpt"))#, step)

	def train(self, X, step):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_raw: X})
		self.train_writer.add_summary(summary, step)
		return 


if __name__ == "__main__":

	data, fs = sf.read('test.flac')
	data = data[0:20*256]
	shape = data.shape
	data = data[np.newaxis, :]
	print data.shape
	ada = Adapt()
	ada.init()
    
	for i in range(1000):
		_ , y = ada.output(data, i)
		print y
        
	# y = np.reshape(y, (195, 256))
	print y.shape
	print np.count_nonzero(y)
