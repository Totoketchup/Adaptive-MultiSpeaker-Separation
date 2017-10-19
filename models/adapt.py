# -*- coding: utf-8 -*-
# My Model 
from utils.ops import ops
from utils.ops.ops import unpool, variable_summaries, get_scope_variable
# from tensorflow.contrib.tensorboard.plugins import projector
# from utils.postprocessing.reconstruction import 
# import matplotlib.pyplot as plt
import os
import config
import tensorflow as tf
import haikunator


name = 'AdaptiveNet'

#############################################
#     Adaptive Front and Back End Model     #
#############################################

class Adapt:
	def __init__(self, config_model=None, N=256, maxpool=256, l=0.001, pretraining=True, runID=None, separator=None, folder=''):
		
		self.N = N
		self.max_pool_value = maxpool
		self.l = l
		self.pretraining = pretraining
		self.folder = folder
		self.sepNet = separator
		self.smooth_size = 4.0
		self.p = 0.05
		self.beta = 1.0

		if config_model != None:
			self.N = config_model['N']
			self.max_pool_value = config_model['maxpool']
			self.l = config_model['reg']
			self.folder = config_model['type']
			self.smooth_size = config_model['smooth_size']
			self.beta = config_model['beta']
			self.p = config_model['rho']
			self.window = config_model['window']
		
		if runID == None:
			# Run ID for tensorboard
			self.runID = name + '-' + haikunator.Haikunator().haikunate()
			print 'ID : {}'.format(self.runID)
			if config_model != None:
				self.runID += ''.join('-{}={}-'.format(key, val) for key, val in sorted(config_model.items()))
		else:
			self.runID = name + '-' + runID


		self.graph = tf.Graph()

		with self.graph.as_default():

			# Batch of raw mixed audio - Input data
			# shape = [ batch size , samples ] = [ B , L ]
			self.X_mix = tf.placeholder("float", [None, None])

			# Speakers indicies used in the mixtures
			# shape = [ batch size, #speakers]
			self.Ind = tf.placeholder(tf.int32, [None,None])

			self.learning_rate = tf.placeholder("float")
			tf.summary.scalar('learning_rate', self.learning_rate)

			shape_in = tf.shape(self.X_mix)
			self.B = shape_in[0]
			self.L = shape_in[1]

			# Network Variables:
			# 	self.W = tf.get_variable("W",shape=[1, 1024, 1, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
			# 	variable_summaries(self.W)

			# with tf.name_scope('smooth'):
			# 	self.smoothing_filter = tf.get_variable("smoothing_filter",shape=[1, int(self.smooth_size), 1, 1], initializer=tf.contrib.layers.xavier_initializer_conv2d())
			# 	variable_summaries(self.smoothing_filter)
			with tf.name_scope('conv'):
				self.W = get_scope_variable('conv', "W", shape=[1, self.window, 1, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
				self.WT = get_scope_variable('deconv', "WT", shape=[1, self.window, 1, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
				variable_summaries(self.W)
			# self.smoothing_filter = get_scope_variable('smooth', "smoothing_filter", shape=[1, int(self.smooth_size), 1, 1], initializer=tf.contrib.layers.xavier_initializer_conv2d())
			with tf.name_scope('smooth'):
				self.smoothing_filter = get_scope_variable('smooth', "smoothing_filter", shape=[1, int(self.smooth_size), self.N, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
				variable_summaries(self.smoothing_filter)
			

			# Batch of raw non-mixed audio
			# shape = [ batch size , number of speakers, samples ] = [ B, S, L]
			self.X_non_mix = tf.placeholder("float", [None, None, None])
			self.shape_non_mix = tf.shape(self.X_non_mix)
			self.S = self.shape_non_mix[1]

			# tf.summary.audio(name= "input/non_mix_1", tensor = self.X_non_mix[:, 0, :], sample_rate = config.fs)
			# tf.summary.audio(name= "input/non_mix_2", tensor = self.X_non_mix[:, 1, :], sample_rate = config.fs)
			# tf.summary.audio(name= "input/mix", tensor = self.X_mix, sample_rate = config.fs)
		
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
			with tf.name_scope('preprocessing'):
				# self.x = tf.concat([self.X_mix,  tf.reshape(self.X_non_mix, [self.B*self.S, self.L])], axis=0)
				self.x = tf.reshape(self.X_non_mix, [self.B*self.S, self.L])
				self.B_tot = self.B*self.S

			if pretraining:
				
				self.front
				self.separator
				self.back
				self.cost
				self.optimize

				# tf.summary.audio(name= "output/separated_1", tensor = self.back[:, 0, :], sample_rate = config.fs)
				# tf.summary.audio(name= "output/separated_2", tensor = self.back[:, 1, :], sample_rate = config.fs)

				# tf.summary.audio(name= "subtracted/1", tensor = self.X_mix - self.back[:, 1, :], sample_rate = config.fs)
				# tf.summary.audio(name= "subtracted/2", tensor = self.X_mix - self.back[:, 0, :], sample_rate = config.fs)
				

			else:
				self.front
			
		# Create a session for this model based on the constructed graph
		config_ = tf.ConfigProto()
		config_.gpu_options.allow_growth = True
		config_.allow_soft_placement = True
		config_.log_device_placement = True
		self.sess = tf.Session(graph=self.graph, config=config_)


	def tensorboard_init(self):
		with self.graph.as_default():
			self.merged = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(os.path.join(config.log_dir,self.folder,self.runID), self.graph)
			self.saver = tf.train.Saver()

		# if self.sepNet != None:
		# 	config_ = projector.ProjectorConfig()
		# 	embedding = config_.embeddings.add()
		# 	embedding.tensor_name = self.sepNet.prediction.name
		# 	projector.visualize_embeddings(self.train_writer, config_)


	def create_saver(self):
		with self.graph.as_default():
			self.saver = tf.train.Saver()

	def restore_model(self, folder, runID):
		self.saver.restore(self.sess, os.path.join(config.log_dir, folder, name+'-'+runID, 'model.ckpt'))


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

		# X = tf.reshape(X, [self.B_tot, -1, self.N, 1])
		# self.X_cost = X[:, 0, :, :]

		# X : [ B_tot , T , N, 1]
		X_abs = tf.abs(X)
		self.T = tf.shape(X_abs)[2]

		# Smoothing the ''STFT-like'' created by the previous OP
		M = tf.nn.conv2d(X_abs, self.smoothing_filter, strides=[1, 1, 1, 1], padding="SAME")
		M = tf.reshape(M, [self.B_tot, -1, self.N, 1])
		X = tf.reshape(X, [self.B_tot, -1, self.N, 1])

		self.p_hat = tf.reduce_sum(X, axis=[1,2,3])/(tf.cast(self.T*self.N, tf.float32))

		M = tf.nn.softplus(M)

		# tf.summary.image(name= "smoothed_spectrogram", tensor = tf.transpose(M, [0,2,1,3]))

		# Matrix for the reconstruction process P = X / M => X = P * M
		# Equivalent to the STFT phase, keep important information for reconstruction (locality)
		# shape = [ B_tot, T , N, 1]
		P = X / M

		# Max Pooling with argmax for unpooling later in the back-end layer
		y, argmax = tf.nn.max_pool_with_argmax(M, (1, self.max_pool_value, 1, 1),
													strides=[1, self.max_pool_value, 1, 1], padding="SAME")
		
		# tf.summary.image(name= "input_spectrogram", tensor = tf.transpose(y, [0,2,1,3]))

		return y, P, argmax


	@ops.scope
	def separator(self):
		# shape = [B_tot, T_, N, 1], shape = [B(1+S), T , N, 1], [B(1+S), T_ , N, 1]
		separator_in, P_in, argmax_in = self.front

		if self.pretraining:
			######################################
			##
			## Signal separator for pretraining 
			##
			######################################
			 
			self.T_max_pooled = tf.shape(separator_in)[1]
		
			return tf.reshape(separator_in,[self.B*self.S, self.T_max_pooled, self.N, 1]), P_in, argmax_in
			
			# # shape = [B*S , 1 , T_, N, 1]
			# self.separator_out = tf.abs(separator_in_mixed - separator_in_added)
			
			# self.separator_out = tf.reshape(self.separator_out, [self.B*self.S, self.T_max_pooled, self.N, 1])
			# return self.separator_out, P_in, argmax_in
		else:
			return self.sepNet.output, P_in, argmax_in

	@ops.scope
	def back(self):
		# Back-End
		input_tensor, P, argmax = self.separator

		# Unpooling (the previous max pooling)
		unpooled = unpool(input_tensor, argmax, ksize=[1, self.max_pool_value, 1, 1], scope='unpool')

		output = unpooled * P

		output = tf.reshape(output, [self.B*self.S, 1, self.T, self.N])
		output = tf.nn.conv2d_transpose(output , filter=self.WT,
									 output_shape=[self.B*self.S, 1, self.L, 1],
									 strides=[1, 1, 1, 1])

		output = tf.reshape(output, [self.B, self.S, self.L])

		return output


	def logfunc(self, x, x2):
		cx = tf.clip_by_value(x, 1e-10, 1.0)
		cx2 = tf.clip_by_value(x2, 1e-10, 1.0)
		return tf.multiply(x, tf.log(tf.div(cx,cx2)))


	def kl_div(self, p, p_hat):
		inv_p = 1 - p
		inv_p_hat = 1 - p_hat 
		return self.logfunc(p, p_hat) + self.logfunc(inv_p, inv_p_hat)


	@ops.scope
	def cost(self):
			# Definition of cost for Adapt model
			# Regularisation
			# shape = [B_tot, T, N]
			self.sparse_reg= self.beta * tf.reduce_mean(self.kl_div(self.p, self.p_hat))
			self.reg = self.l * (tf.nn.l2_loss(self.W) +tf.nn.l2_loss(self.WT))
			
			# input_shape = [B, S, L]
			# Doing l2 norm on L axis : 
			self.cost_1 = 0.5 * tf.reduce_sum(tf.pow(self.X_non_mix - self.back, 2), axis=2) / tf.cast(self.L, tf.float32)

			# shape = [B, S]
			# Compute mean over the speakers
			cost = tf.reduce_mean(self.cost_1, 1)
			# cost = self.cost_1
			# shape = [B]
			# Compute mean over batches
			MSE = tf.reduce_mean(cost, 0) 
			cost = MSE + self.sparse_reg + self.reg

			tf.summary.scalar('loss', MSE)
			tf.summary.scalar('sparsity', tf.reduce_mean(self.p_hat))
			tf.summary.scalar('sparse_reg', self.sparse_reg)
			tf.summary.scalar('regularization', self.reg)
			tf.summary.scalar('training_cost', cost)
			return cost




	@ops.scope
	def optimize(self):
		if hasattr(self, 'trainable_variables') == False:
			self.trainable_variables = tf.global_variables()
		print self.trainable_variables
		return tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, var_list=self.trainable_variables)


	def save(self, step):
		self.saver.save(self.sess, os.path.join(config.log_dir,self.folder ,self.runID, "model.ckpt"))  # , step)

	@staticmethod
	def load(config_model, runID, pretraining ,folder='', new_folder=''):
		adapt = Adapt(config_model=config_model,pretraining= pretraining, folder=new_folder)
		adapt.saver.restore(adapt.sess, os.path.join(config.log_dir, folder, name+'-'+runID, 'model.ckpt'))
		return adapt

	def train(self, X_mix, X_in, learning_rate, step, ind_train=None):
		if ind_train is None:
			summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in, self.learning_rate:learning_rate})
		else:
			summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in, self.Ind:ind_train, self.learning_rate:learning_rate})

		self.train_writer.add_summary(summary, step)
		return cost

	def train_no_sum(self, X_mix, X_in, learning_rate, step, ind_train=None):
		if ind_train is None:
			_, cost = self.sess.run([self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in, self.learning_rate:learning_rate})
		else:
			_, cost = self.sess.run([self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in, self.Ind:ind_train, self.learning_rate:learning_rate})
		return cost

	def connect_front(self, separator_class):
		# Separate Mixed and Non Mixed 'spectrograms'
		with tf.name_scope('split_front'):
			X = tf.reshape(self.front[0][:self.B, :, :], [self.B, -1, self.N])

			X_non_mix = tf.reshape(self.front[0][self.B:, :, :, :], [self.B, self.S, -1, self.N])
			X_non_mix = tf.transpose(X_non_mix, [0,2,3,1])

		self.sepNet = separator_class((X, X_non_mix), self)

	def connect_back(self):
		self.back

	def freeze_front(self):
		training_var = tf.trainable_variables()
		training_var.remove(self.W)
		training_var.remove(self.smoothing_filter)
		self.trainable_variables = training_var
