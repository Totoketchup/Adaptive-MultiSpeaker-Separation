# My Model 
from utils.ops import ops
from utils.ops.ops import Residual_Net, Conv1D, Reshape, Dense, unpool, f_props
from tensorflow.contrib.tensorboard.plugins import projector
# from utils.postprocessing.reconstruction import 
# import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import config
import tensorflow as tf
import time
import soundfile as sf
import numpy as np


#############################################
#     Adaptive Front and Back End Model     #
#############################################

class Adapt:
	def __init__(self, pretraining=True):
		self.N = 256
		self.max_pool_value = self.N
		self.pretraining = True
		self.graph = tf.Graph()

		with self.graph.as_default():
			# Batch of raw mixed audio - Input data
			# shape = [ batch size , samples ]
			self.X_raw = tf.placeholder("float", [None, None])

			# Batch of raw non-mixed audio
			# shape = [ batch size , number of speakers, samples ]
			self.X_in = tf.placeholder("float", [None, 2, None])

			# Network Variables:
			self.WC1 = tf.Variable(tf.random_normal([1024, 1, self.N], stddev=0.35))
			self.smoothing_filter = tf.Variable(tf.random_normal([1, 4, 1, 1], stddev=0.35))


			self.front
			self.separator
			self.back
			self.cost
			
			self.optimize

			self.saver = tf.train.Saver()
			self.merged = tf.summary.merge_all()

			self.train_writer = tf.summary.FileWriter('log/adaptive', self.graph)

		# Create a session for this model based on the constructed graph
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		self.sess = tf.Session(graph=self.graph, config=config)

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
		in_mix = tf.expand_dims(self.X_raw, 2)
		self.input_shape = tf.shape(self.X_raw)

		output = self.front_func(in_mix)

		if self.pretraining:
			in_raw =  tf.expand_dims(tf.transpose(self.X_in, [1,0,2]), 3)
			y_raw, _, _  = tf.map_fn(self.front_func, in_raw, swap_memory=True, dtype=(tf.float32, tf.float32, tf.int64))

			return output, y_raw

		return output


	def front_func(self, input_tensor):
		# 1 Dimensional convolution along T axis with a window length = 1024
		# And N = 256 filters
		X = tf.nn.conv1d(input_tensor, self.WC1, stride=1, padding="SAME")

		# X : [ B , T , N]
		X_abs = tf.abs(X)

		# Smoothing the 'STFT' like created by the previous OP
		M = tf.nn.conv2d(tf.expand_dims(X_abs, 3), self.smoothing_filter, strides=[1, 1, 1, 1], padding="SAME")

		M = tf.nn.softplus(M)

		# Matrix for the reconstruction process P = M / X => X * P = M
		P = M / tf.expand_dims(X, 3)

		# Max Pooling
		# with tf.device('/GPU'):
		y, argmax = tf.nn.max_pool_with_argmax(M, (1, self.max_pool_value, 1, 1),
													strides=[1, self.max_pool_value, 1, 1], padding="SAME")

		with tf.device("/cpu:0"):
			a = tf.identity(argmax)
		
		return y, P, a

	@ops.scope
	def separator(self):
		## ##
		## Signal separator network for testing.
		## ##
		if self.pretraining:
			(separator_in, P_in, argmax_in), separator_in_raw = self.front
			self.separator_out = tf.expand_dims(separator_in,0) - separator_in_raw 
			shapu = tf.shape(self.X_in)[1]
			return self.separator_out, tf.tile(tf.expand_dims(P_in, 0),[shapu,1,1,1,1]), tf.tile(tf.expand_dims(argmax_in, 0),[shapu,1,1,1,1])
		else:	
			separator_input, P, argmax = self.front
			separator_out = self.separator_func(separator_input)
			return separator_out_raw, P_raw, argmax_raw

		# PRETRAINING SEPARATOR

	def separator_func(self, tensor_input):

		input = tf.squeeze(tensor_input, [3])
		
		shape = tf.shape(input)
		layers = [
			Dense(self.N, 500, tf.nn.softplus),
			Dense(500, 500, tf.nn.softplus),
			Dense(500, self.N, tf.nn.softplus)
		]

		separator_out = f_props(layers, tf.reshape(input, [-1, self.N]))
		return separator_out


	@ops.scope
	def back(self):
		# Back-End
		if self.pretraining:
			return tf.map_fn(self.back_func, self.separator, swap_memory=True, dtype=tf.float32)
		else:
			return back_func(self.separator)


	def back_func(self, input_tensor):
		# Back-End
		back_input, P, argmax = input_tensor

		# Unpooling (the previous max pooling)
		unpooled = unpool(back_input, argmax, ksize=[1, self.max_pool_value, 1, 1], scope='unpool')

		out = unpooled * P

		out = tf.reshape(out, [self.input_shape[0], self.input_shape[1], self.N, 1])
		out = tf.nn.conv2d_transpose(tf.transpose(out, [0, 1, 3, 2]), filter=tf.expand_dims(self.WC1, 0),
									 output_shape=[self.input_shape[0], self.input_shape[1], 1, 1],
									 strides=[1, 1, 1, 1])

		return tf.reshape(out, self.input_shape)

	@ops.scope
	def cost(self):
		# Definition of cost for Adapt model
		self.reg = 0.001 * tf.norm(self.X_raw, axis=1)
		if self.pretraining:
			cost = tf.map_fn(self.cost_func, (tf.transpose(self.X_in, [1,0,2]), self.back), dtype=tf.float32)
			cost = tf.reduce_mean(cost, 0)
		else:
			# TODO
			# cost = tf.reduce_sum(tf.pow(X_in - X_reconstruct, 2), axis=1) + self.reg
			cost = tf.reduce_mean(cost)
		tf.summary.scalar('training cost', cost)
		return cost


	def cost_func(self, input_tensor):
		X_in, X_reconstruct = input_tensor
		# Definition of cost for Adapt model
		cost = tf.reduce_sum(tf.pow(X_in - X_reconstruct, 2), axis=1) + self.reg
		cost = tf.reduce_mean(cost)
		return cost
	@ops.scope
	def optimize(self):
		return tf.train.AdamOptimizer().minimize(self.cost)

	def save(self, step):
		self.saver.save(self.sess, os.path.join('log/adaptive/', "adaptive_model.ckpt"))  # , step)

	def train(self, X_mix,X_in, step):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_raw: X_mix, self.X_in:X_in})
		self.train_writer.add_summary(summary, step)
		return cost

	def test(self, X, X_in):
		cost = self.sess.run(self.cost, {self.X_raw: X, self.X_in: X_in})
		return cost


if __name__ == "__main__":
	N = 256
	batch_size = 10
	length = 5*512

	data, fs = sf.read('test.flac')
	sub = len(data)/length
	print 'SUB = ', sub
	data = data[0:len(data) - len(data) % (N * sub)]
	data = np.array(np.split(data, sub))
	shape = data.shape
	# data = data[np.newaxis, :]
	ada = Adapt()
	ada.init()

	for u in range(1):
		for i in range(sub / batch_size):
			d = data[i*batch_size:(i + 1) * batch_size, :]
			d_in = np.transpose(np.array([d,d]),(1,0,2))
			y = ada.train(d, d_in, (sub / batch_size) * u + i)
			# y = ada.test(d, d_in)
			print y

	X_reconstruct = ada.train(data, data, 0)
	sf.write('test_recons.flac', np.flatten(X_reconstruct), fs)

	# y = np.reshape(y, (195, 256))
	print y.shape
	print np.count_nonzero(y)
