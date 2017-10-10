# -*- coding: utf-8 -*-
# My Model 
from utils.ops import ops
from utils.ops.ops import Residual_Net, Conv1D, Reshape, Dense, unpool, f_props, variable_summaries
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
import haikunator


name = 'AdaptiveNet'

#############################################
#     Adaptive Front and Back End Model     #
#############################################

class Adapt:
	def __init__(self, pretraining=True, runID=None, separator=None):
		self.N = 256
		self.max_pool_value = 128
		self.l = 0.001
		self.pretraining = pretraining
		
		if separator != None:
			self.sepNet = separator
		

		if runID == None:
			# Run ID for tensorboard
			self.runID = name + '-' + haikunator.Haikunator().haikunate()
			print 'ID : {}'.format(self.runID)
		else:
			self.runID = name + '-' + runID


		self.graph = tf.Graph()

		with self.graph.as_default():

			# Batch of raw mixed audio - Input data
			# shape = [ batch size , samples ] = [ B , L ]
			self.X_mix = tf.placeholder("float", [None, None])

			self.learning_rate = tf.placeholder("float")
			tf.summary.scalar('learning rate', self.learning_rate)

			shape_in = tf.shape(self.X_mix)
			self.B = shape_in[0]
			self.L = shape_in[1]

						# Network Variables:
			with tf.name_scope('conv'):
				self.W = tf.get_variable("W",shape=[1, 1024, 1, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
				variable_summaries(self.W)

			with tf.name_scope('smooth'):
				self.filter_size = 4.0
				self.smoothing_filter = tf.get_variable("smoothing_filter",shape=[1, int(self.filter_size), 1, 1], initializer=tf.contrib.layers.xavier_initializer_conv2d())
				variable_summaries(self.smoothing_filter)
			

			# Batch of raw non-mixed audio
			# shape = [ batch size , number of speakers, samples ] = [ B, S, L]
			self.X_non_mix = tf.placeholder("float", [None, None, None])
			self.shape_non_mix = tf.shape(self.X_non_mix)
			self.S = self.shape_non_mix[1]

			tf.summary.audio(name= "input/non_mix_1", tensor = self.X_non_mix[:, 0, :], sample_rate = config.fs)
			tf.summary.audio(name= "input/non_mix_2", tensor = self.X_non_mix[:, 1, :], sample_rate = config.fs)
			tf.summary.audio(name= "input/mix", tensor = self.X_mix, sample_rate = config.fs)

		
			# The following part is doing: (example with 4 elements) 
			# We have : X_mixed = [a+b+c+d]
			#
			# input : X_non_mix = [ a, b, c, d]
			# output : Y = [b+c+d, a+c+d, a+b+d, a+b+c]
			# 
			# With broadcasting at the end of the front end layer, naming F(x) the front end function
			# We can compute: D = F(X_mix) - F(Y) equivalent to perfect separation for each signal from
			# the mixed input.
			# Rolling test	
			init = tf.concat([tf.slice(self.X_non_mix,[0, self.S-1, 0], [-1, 1, -1]),
				tf.slice(self.X_non_mix,[0, 0, 0], [-1, self.S-1, -1])], axis=1)	

			i = (tf.constant(1), init)
			cond = lambda i, x: tf.less(i, self.S-1)
			body = lambda i, x: (i + 1, x + tf.concat([tf.slice(x,[0, self.S-1, 0], [-1, 1, -1]),
				tf.slice(x,[0, 0, 0], [-1, self.S-1, -1])], axis=1))
			_, X_added_signal = tf.while_loop(cond, body, i)

			# Shape added signal = [B*S, L]
			X_added_signal = tf.reshape(X_added_signal, [self.B*self.S, self.L])

			# shape = [ B*(1 + S), L ]
			self.B_tot = self.B*(self.S+1)
			self.x = tf.concat([self.X_mix, X_added_signal], axis=0)

			if pretraining:
				
				self.front
				self.separator
				self.back
				self.cost
				self.optimize

				tf.summary.audio(name= "output/separated_1", tensor = self.back[:, 0, :], sample_rate = config.fs)
				tf.summary.audio(name= "output/separated_2", tensor = self.back[:, 1, :], sample_rate = config.fs)

			else:
				self.front
				if separator != None:
					self.sepNet = separator(self.front[0], self.graph,B=self.B, S=self.S, F=self.N, E=config.embedding_size, threshold=config.threshold)
				self.separator
				self.cost
				self.optimize



			self.image_spectro = tf.summary.image(name= "spectrogram", tensor = self.front[0])
			self.saver = tf.train.Saver([self.W, self.smoothing_filter])
			self.merged = tf.summary.merge_all()

			self.train_writer = tf.summary.FileWriter('log/'+self.runID, self.graph)

		# Create a session for this model based on the constructed graph
		config_ = tf.ConfigProto()
		config_.gpu_options.allow_growth = True
		config_.allow_soft_placement = True
		self.sess = tf.Session(graph=self.graph, config=config_)

	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	##
	## Front End creating STFT like data + P Matrix
	##
	@ops.scope
	def front(self):
		# Front-End

		# x : [ B-tot , 1, L , 1]
		input_front =  tf.expand_dims(tf.expand_dims(self.x, 2), 1)

		# 1 Dimensional convolution along T axis with a window length = 1024
		# And N = 256 filters
		X = tf.nn.conv2d(input_front, self.W, strides=[1, 1, 1, 1], padding="SAME")

		X = tf.reshape(X, [self.B_tot, -1, self.N, 1])
		self.X_cost = X[:, :, :, 0]

		# X : [ B_tot , T , N, 1]
		X_abs = tf.abs(X)
		self.T = tf.shape(X_abs)[1]

		# Smoothing the ''STFT-like'' created by the previous OP
		M = tf.nn.conv2d(X_abs, self.smoothing_filter, strides=[1, 1, 1, 1], padding="SAME")

		M = tf.nn.softplus(M)

		# Matrix for the reconstruction process P = X / M => X = P * M
		# Equivalent to the STFT phase, keep important information for reconstruction
		# shape = [ B_tot, T , N, 1]
		P = X / M

		# Max Pooling with argmax for unpooling later in the back-end layer
		y, argmax = tf.nn.max_pool_with_argmax(M, (1, self.max_pool_value, 1, 1),
													strides=[1, self.max_pool_value, 1, 1], padding="SAME")
		
		return y, P, argmax


	@ops.scope
	def separator(self):

		if self.pretraining:
			######################################
			##
			## Signal separator for pretraining 
			##
			######################################

			# shape = [B_tot, T_, N, 1], shape = [B(1+S), T , N, 1], [B(1+S), T_ , N, 1]
			separator_in, P_in, argmax_in = self.front
			self.T_max_pooled = tf.shape(separator_in)[1]

			# shape = [ B, T_, N , 1]
			separator_in_mixed = separator_in[:self.B, :, : , :]
			separator_in_added = separator_in[self.B:, :, : , :]

			# shape = [ B, T_, N , 1]
			P_in = P_in[:self.B, :, : , :]
			argmax_in = argmax_in[:self.B, :, : , :]

			argmax_in = tf.tile(argmax_in, [self.S,1,1,1]) 
			P_in = tf.tile(P_in, [self.S,1,1,1])
			
			# shape = [ B, 1, T_, N, 1]
			separator_in_mixed = tf.expand_dims(separator_in_mixed, 1)

			# shape = [ B, S, T_, N, 1]
			separator_in_added = tf.reshape(separator_in_added, [self.B, self.S, self.T_max_pooled, self.N, 1])

			# shape = [B*S , 1 , T_, N, 1]
			self.separator_out = separator_in_mixed - separator_in_added
			
			self.separator_out = tf.reshape(self.separator_out, [self.B*self.S, self.T_max_pooled, self.N, 1])
			return self.separator_out, P_in, argmax_in
		else:
			#TODO not finished, have to load a network for the separation task
			return self.sepNet.prediction

	@ops.scope
	def back(self):
		# Back-End
		if self.pretraining:
			input_tensor, P, argmax = self.separator

			# layers = [
			# 	Dense(self.N, self.N)
			# ]

			# adj = f_props(layers, tf.reshape(back_input, [-1, self.N]))
			# back_input = tf.reshape(adj,[self.B*self.S, self.T_max_pooled, self.N, 1])

			# Unpooling (the previous max pooling)
			unpooled = unpool(input_tensor, argmax, ksize=[1, self.max_pool_value, 1, 1], scope='unpool')

			output = unpooled * P

			output = tf.reshape(output, [self.B*self.S, 1, self.T, self.N])
			output = tf.nn.conv2d_transpose(output , filter=self.W,
										 output_shape=[self.B*self.S, 1, self.L, 1],
										 strides=[1, 1, 1, 1])

			return tf.reshape(output, [self.B, self.S, self.L])
		else:
			return None

	@ops.scope
	def cost(self):
		if self.pretraining:
			# Definition of cost for Adapt model

			# Regularisation
			# shape = [ B_tot, T, N]
			self.reg = tf.norm(self.X_cost, ord=1, axis=[1 ,2]) # norm 1
			self.reg = self.l*tf.reduce_mean(self.reg, 0) # mean over batches

			# input_shape = [B, S, L]
			# Doing l2 norm on L axis : 
			self.cost_1 = tf.reduce_sum(tf.pow(self.X_non_mix - self.back, 2), axis=2)
			# shape = [B, S]
			# Compute mean over the speakers
			cost = tf.reduce_mean(self.cost_1, 1)

			# shape = [B]
			# Compute mean over batches
			MSE = tf.reduce_mean(cost, 0) 
			cost = MSE + self.reg

			tf.summary.scalar('MSE loss', MSE)
			tf.summary.scalar('Regularization', self.reg)
			tf.summary.scalar('Training cost', cost)
			return cost
		else:
			return self.sepNet.cost



	@ops.scope
	def optimize(self):
		return tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

	def save(self, step):
		self.saver.save(self.sess, os.path.join('log_model/', self.runID + "-model.ckpt"))  # , step)

	@staticmethod
	def load(runID, pretraining, separator=None):
		adapt = Adapt(pretraining=pretraining, runID=None, separator=separator)
		adapt.init()
		adapt.saver.restore(adapt.sess, config.model_dir + '/' + name + '-' + runID + "-model.ckpt")
		return adapt

	def train(self, X_mix, X_in, learning_rate, step):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in, self.learning_rate:learning_rate})
		self.train_writer.add_summary(summary, step)
		return cost

	def test(self, X, X_in):
		cost = self.sess.run(self.testoune, {self.X_mix: X, self.X_non_mix: X_in})
		return cost