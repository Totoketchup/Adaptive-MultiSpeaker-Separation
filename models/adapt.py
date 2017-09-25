# -*- coding: utf-8 -*-
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
			# shape = [ batch size , samples ] = [ B , L ]
			self.X_mix = tf.placeholder("float", [None, None])

			if pretraining:
				# Batch of raw non-mixed audio
				# shape = [ batch size , number of speakers, samples ] = [ B, S , L]
				self.X_non_mix = tf.placeholder("float", [None, None, None])
				self.X_non_mix_t = tf.transpose(self.X_non_mix, [1, 0, 2])
				self.shape_non_mix = tf.shape(self.X_non_mix_t)
				self.S = self.shape_non_mix[0]
				self.B = self.shape_non_mix[1]
				self.L = self.shape_non_mix[2]

				# shape = [ S*B , L ]
				self.X_in_ = tf.reshape(self.X_non_mix_t, [self.B*self.S, self.L])

				# shape = [ B*(1 + S), L ] = [ B + B*S , L]
				self.x = tf.concat([self.X_mix, self.X_in_], axis=0)
			else:
				self.x = self.X_mix

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

		# x : [ B or B*(1+S) , L , 1]
		in_mix = tf.expand_dims(self.x, 2)
		self.front_input_shape = tf.shape(self.x)

		output = self.front_func(in_mix)
		
		return output


	def front_func(self, input_tensor):
		# 1 Dimensional convolution along T axis with a window length = 1024
		# And N = 256 filters
		X = tf.nn.conv1d(input_tensor, self.WC1, stride=1, padding="SAME")

		# X : [ B or B(1+S) , T , N]
		X_abs = tf.abs(X)
		self.T = tf.shape(X_abs)[1]

		# Smoothing the 'STFT' like created by the previous OP
		M = tf.nn.conv2d(tf.expand_dims(X_abs, 3), self.smoothing_filter, strides=[1, 1, 1, 1], padding="SAME")

		M = tf.nn.softplus(M)

		# Matrix for the reconstruction process P = M / X => X * P = M
		# shape = [ B or B(1+S), T , N, 1]
		P = M / tf.expand_dims(X, 3)
		# Max Pooling
		# with tf.device('/GPU'):
		y, argmax = tf.nn.max_pool_with_argmax(M, (1, self.max_pool_value, 1, 1),
													strides=[1, self.max_pool_value, 1, 1], padding="SAME")
		
		return y, P, argmax

	@ops.scope
	def separator(self):
		## ##
		## Signal separator network for testing.
		## ##
		if self.pretraining:
			# shape = [B(1+S), T_, N, 1], shape = [B(1+S), T , N, 1], [B(1+S), T_ , N, 1]
			separator_in, P_in, argmax_in = self.front
			self.T_p = tf.shape(separator_in)[1]

			# Splitting Mixed and Non Mixed data , from (B + B*S, ..) to ((B , ..), (B*S, ...) 
			split = [self.B, self.B*self.S]
			print separator_in
			# shape = [ B, T_, N , 1], shape = [ B*S, T_, N , 1]
			separator_in_mixed, separator_in_non_mixed  = tf.split(separator_in, split, axis=0)
			# shape = [ B, T_, N , 1]
			P_in, _  = tf.split(P_in, split)
			argmax_in, _  = tf.split(argmax_in, split)

			# shape = [ B, 1, T_, N, 1]
			separator_in_mixed = tf.expand_dims(separator_in_mixed, 1)
			shape = [self.B, self.S, self.T_p, self.N, 1]
			separator_in_non_mixed = tf.reshape(separator_in_non_mixed, shape)
			# shape = [B , S , T_, N, 1]
			self.separator_out = separator_in_mixed - separator_in_non_mixed 
			
			self.separator_out = tf.reshape(self.separator_out, [self.B*self.S, self.T_p, self.N, 1])
			return self.separator_out, P_in, argmax_in
		else:
			return None

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
		back_in, P_in, argmax_in = self.separator

		argmax_in = tf.tile(argmax_in, [self.S,1,1,1])
		P_in = tf.tile(P_in, [self.S,1,1,1])

		return self.back_func((back_in, P_in, argmax_in))


	def back_func(self, input_tensor):
		# Back-End
		back_input, P, argmax = input_tensor

		# Unpooling (the previous max pooling)
		unpooled = unpool(back_input, argmax, ksize=[1, self.max_pool_value, 1, 1], scope='unpool')

		out = unpooled * P

		out = tf.reshape(out, [self.B*self.S, self.T, self.N, 1])
		out = tf.nn.conv2d_transpose(tf.transpose(out, [0, 1, 3, 2]), filter=tf.expand_dims(self.WC1, 0),
									 output_shape=[self.B*self.S, self.L, 1, 1],
									 strides=[1, 1, 1, 1])

		return tf.reshape(out, [self.B, self.S, self.L])

	@ops.scope
	def cost(self):
		# Definition of cost for Adapt model
		# shape = [B, 1]
		self.reg = 0.001 * tf.norm(self.X_mix, axis=1)

		#input_shape = [B, S, L]
		if self.pretraining:
			# shape = [B, S]
			# Doing l2 norm on L axis
			cost = tf.reduce_sum(tf.pow(self.X_non_mix - self.back, 2), axis=2)
			# shape = [B]
			# Compute mean over the speakers
			cost = tf.reduce_mean(cost, 1) #+ self.reg
			# shape = ()
			# Compute mean over batches
			cost = tf.reduce_mean(cost, 0)
		else:
			# TODO
			# cost = tf.reduce_sum(tf.pow(X_in - X_reconstruct, 2), axis=1) + self.reg
			cost = tf.reduce_mean(cost)

		tf.summary.scalar('training cost', cost)
		return cost/1e5


	@ops.scope
	def optimize(self):
		return tf.train.AdamOptimizer().minimize(self.cost)

	def save(self, step):
		self.saver.save(self.sess, os.path.join('log/adaptive/', "adaptive_model.ckpt"))  # , step)

	def train(self, X_mix,X_in, step):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in})
		self.train_writer.add_summary(summary, step)
		return cost

	def test(self, X, X_in):
		cost = self.sess.run(self.cost, {self.X_mix: X, self.X_non_mix: X_in})
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

	for u in range(100):
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
