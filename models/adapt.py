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
	def __init__(self, pretraining=True, runID=None):
		self.N = 256
		self.max_pool_value = 128
		self.pretraining = True
		self.graph = tf.Graph()
		

		if runID == None:
			# Run ID for tensorboard
			self.runID = name + '-' + haikunator.Haikunator().haikunate()
			print 'ID : {}'.format(self.runID)
		else:
			self.runID = runID

		with self.graph.as_default():
			# Batch of raw mixed audio - Input data
			# shape = [ batch size , samples ] = [ B , L ]
			self.X_mix = tf.placeholder("float", [None, None])
			self.learning_rate = tf.placeholder("float")
			tf.summary.scalar('learning rate', self.learning_rate)

			if pretraining:
				# Batch of raw non-mixed audio
				# shape = [ batch size , number of speakers, samples ] = [ B, S, L]
				self.X_non_mix = tf.placeholder("float", [None, None, None])

				self.shape_non_mix = tf.shape(self.X_non_mix)
				self.B = self.shape_non_mix[0]
				self.S = self.shape_non_mix[1]
				self.L = self.shape_non_mix[2]

				# Rolling test		
				init = tf.concat([tf.slice(self.X_non_mix,[0, self.S-1, 0], [-1, 1, -1]),
					tf.slice(self.X_non_mix,[0, 0, 0], [-1, self.S-1, -1])], axis=1)	

				i = (tf.constant(1), init)
				cond = lambda i, x: tf.less(i, self.S-1)
				body = lambda i, x: (i + 1, x + tf.concat([tf.slice(x,[0, self.S-1, 0], [-1, 1, -1]),tf.slice(x,[0, 0, 0], [-1, self.S-1, -1])], axis=1))
				_, X_added_signal = tf.while_loop(cond, body, i)

				# Shape added signal = [B*S, L]
				X_added_signal = tf.reshape(X_added_signal, [self.B*self.S, self.L])

				# shape = [ B*(1 + S), L ]
				self.B_tot = self.B*(self.S+1)
				self.x = tf.concat([self.X_mix, X_added_signal], axis=0)
			else:
				self.x = self.X_mix

			# Network Variables:
			with tf.name_scope('conv'):
				self.WC1 = tf.get_variable("WC1",shape=[1, 1024, 1, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
				variable_summaries(self.WC1)

			with tf.name_scope('smooth'):
				self.filter_size = 4.0
				self.smoothing_filter = tf.get_variable("smooth",shape=[1, int(self.filter_size), 1, 1], initializer=tf.contrib.layers.xavier_initializer_conv2d())
				# self.smoothing_filter = tf.Variable(np.full((1,int(self.filter_size),1,1), 1.0/self.filter_size, dtype = np.float32))
				variable_summaries(self.smoothing_filter)

			self.front
			self.separator
			self.back
			self.cost
			self.optimize

			tf.summary.audio(name= "input/mix", tensor = self.X_mix, sample_rate = config.fs)
			tf.summary.audio(name= "input/non_mix_1", tensor = self.X_non_mix[:, 0, :], sample_rate = config.fs)
			tf.summary.audio(name= "input/non_mix_2", tensor = self.X_non_mix[:, 1, :], sample_rate = config.fs)
			tf.summary.audio(name= "output/separated_1", tensor = self.back[:, 0, :], sample_rate = config.fs)
			tf.summary.audio(name= "output/separated_2", tensor = self.back[:, 1, :], sample_rate = config.fs)

			self.image_spectro = tf.summary.image(name= "spectrogram", tensor = self.front[0])
			self.saver = tf.train.Saver()
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
		# Front-End Adapt

		# x : [ B or B*(1+S) , 1, L , 1]
		input_front =  tf.expand_dims(tf.expand_dims(self.x, 2), 1)

		output = self.front_func(input_front)

		return output


	def front_func(self, input_tensor):
		# 1 Dimensional convolution along T axis with a window length = 1024
		# And N = 256 filters
		X = tf.nn.conv2d(input_tensor, self.WC1, strides=[1, 1, 1, 1], padding="SAME")

		X = tf.reshape(X, [self.B_tot, -1, self.N, 1])
		self.X_cost = X[:, :, :, 0]

		# X : [ B or self.length , T , N]
		X_abs = tf.abs(X)
		self.T = tf.shape(X_abs)[1]

		# Smoothing the 'STFT' like created by the previous OP
		M = tf.nn.conv2d(X_abs, self.smoothing_filter, strides=[1, 1, 1, 1], padding="SAME")

		M = tf.nn.softplus(M)

		# Matrix for the reconstruction process P = X / M => X = P * M
		# shape = [ B or self.length, T , N, 1]
		P = X / M

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
			# shape = [B*(S+1), T_, N, 1], shape = [B(1+S), T , N, 1], [B(1+S), T_ , N, 1]
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
			return None

	# # PRETRAINING SEPARATOR
	# def separator_func(self, tensor_input):

	# 	input = tf.squeeze(tensor_input, [3])
		
	# 	shape = tf.shape(input)
	# 	layers = [
	# 		Dense(self.N, 500, tf.nn.softplus),
	# 		Dense(500, 500, tf.nn.softplus),
	# 		Dense(500, self.N, tf.nn.softplus)
	# 	]

	# 	separator_out = f_props(layers, tf.reshape(input, [-1, self.N]))
	# 	return separator_out


	@ops.scope
	def back(self):
		# Back-End
		back_in, P_in, argmax_in = self.separator

		# argmax_in = tf.tile(argmax_in, [self.S,1,1,1])
		# P_in = tf.tile(P_in, [self.S,1,1,1])

		return self.back_func((back_in, P_in, argmax_in))


	def back_func(self, input_tensor):
		# Back-End
		back_input, P, argmax = input_tensor

		# layers = [
		# 	Dense(self.N, self.N)
		# ]

		# adj = f_props(layers, tf.reshape(back_input, [-1, self.N]))
		# back_input = tf.reshape(adj,[self.B*self.S, self.T_max_pooled, self.N, 1])

		# Unpooling (the previous max pooling)
		unpooled = unpool(back_input, argmax, ksize=[1, self.max_pool_value, 1, 1], scope='unpool')

		out = unpooled * P

		out = tf.reshape(out, [self.B*self.S, 1, self.T, self.N])
		out = tf.nn.conv2d_transpose(out , filter=self.WC1,
									 output_shape=[self.B*self.S, 1, self.L, 1],
									 strides=[1, 1, 1, 1])

		a =  tf.reshape(out, [self.B, self.S, self.L])
		# mean, std = tf.nn.moments(a, axes=[2], keep_dims=True)
		# a = (a - mean)/std
		# # a = -1 + 2*(a - tf.reduce_min(a))/(tf.reduce_max(a)-tf.reduce_min(a))
		return a

	@ops.scope
	def cost(self):
		# Definition of cost for Adapt model
		# shape = [B*S, T_, N]
		self.reg = tf.norm(self.X_cost, ord=1, axis=[1 ,2]) # norm 1
		self.reg = 0.01*tf.reduce_mean(self.reg, 0)

		if self.pretraining:
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
			# shape = ()
		else:
			# TODO
			# cost = tf.reduce_sum(tf.pow(X_in - X_reconstruct, 2), axis=1) + self.reg
			cost = tf.reduce_mean(cost)

		tf.summary.scalar('MSE loss', MSE)
		tf.summary.scalar('Norm regulizer', self.reg)
		tf.summary.scalar('training cost', cost)
		return cost


	@ops.scope
	def optimize(self):
		return tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

	def save(self, step):
		self.saver.save(self.sess, os.path.join('log_model/', self.runID+"-model.ckpt"))  # , step)

	@staticmethod
	def load(runID, pretraining=True):
		adapt = Adapt(pretraining=pretraining, runID=runID)
		adapt.saver.restore(adapt.sess, config.model_dir + '/' + name + '-' + runID + "-model.ckpt")
		return adapt

	def train(self, X_mix, X_in, learning_rate, step):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in, self.learning_rate:learning_rate})
		self.train_writer.add_summary(summary, step)
		return cost

	def test(self, X, X_in):
		cost = self.sess.run(self.testoune, {self.X_mix: X, self.X_non_mix: X_in})
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
			# y = ada.train(d, d_in, (sub / batch_size) * u + i)
			y = ada.test(d, d_in)
			print y

	X_reconstruct = ada.train(data, data, 0)
	sf.write('test_recons.flac', np.flatten(X_reconstruct), fs)

	# y = np.reshape(y, (195, 256))
	print y.shape
	print np.count_nonzero(y)
