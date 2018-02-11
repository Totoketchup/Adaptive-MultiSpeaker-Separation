# -*- coding: utf-8 -*-
from utils.ops import BLSTM, Conv1D, Reshape, Normalize, f_props, scope, AMSGrad
import haikunator
from models.Kmeans_2 import KMeans
import os
import config
import tensorflow as tf
import json
from itertools import permutations, compress
############################################
#       Deep Clustering Architecture       #
############################################

class DPCL:

	def __init__(self, adapt=None, **kwargs):

		self.args = kwargs

		if adapt is not None:
			self.layer_size = kwargs['layer_size']
			self.embedding_size = kwargs['embedding_size']
			self.nonlinearity = kwargs['nonlinearity']
			self.normalize = kwargs['normalize']
			
			self.B = adapt.B
			self.S = adapt.S
			self.F = adapt.N

			self.graph = adapt.graph

			with self.graph.as_default():

				with tf.name_scope('split_front'):
					self.X = tf.reshape(adapt.front[0][:self.B, :, :], [self.B, -1, self.F]) # Mix input [B, T, N]
					# Non mix input [B, T, N, S]
					self.X_non_mix = tf.transpose(tf.reshape(adapt.front[0][self.B:, :, :, :], [self.B, self.S, -1, self.F]), [0,2,3,1])

				with tf.name_scope('create_masks'):
					# # Batch of Masks (bins label)
					# # shape = [ batch size, T, F, S]
					argmax = tf.argmax(tf.abs(self.X_non_mix), axis=3)
					self.Y = tf.one_hot(argmax, 2, 1.0, 0.0)
					self.y_test_export = tf.reshape(self.Y[:, :, :, 0], [self.B, -1])

				self.normalization01
				self.prediction
		else:

			#Create a graph for this model
			self.graph = tf.Graph()

			with self.graph.as_default():
				# Global params
				self.folder = kwargs['type']

				# STFT hyperparams
				self.window_size = kwargs['window_size']
				self.hop_size = kwargs['hop_size']

				# Network hyperparams
				self.F = kwargs['freqs_size']
				self.layer_size = kwargs['layer_size']
				self.embedding_size = kwargs['embedding_size']
				self.normalize = kwargs['normalize']
				self.learning_rate = kwargs['learning_rate']
				self.S = kwargs['nb_speakers']

				# Run ID for tensorboard
				self.runID = haikunator.Haikunator().haikunate()
				print 'ID : {}'.format(self.runID)

				# Placeholder tensor for the mixed signals
				self.x_mix = tf.placeholder("float", [None, None])

				# Placeholder tensor for non mixed input data [B, T, F, S]
				# Place holder for non mixed signals [B, S, L]
				self.x_non_mix = tf.placeholder("float", [None, None, None])

				self.preprocessing
				self.normalization01
				self.prediction

				if 'enhance' not in self.folder:
					self.cost
					self.optimize

				config_ = tf.ConfigProto()
				config_.gpu_options.allow_growth = True
				config_.allow_soft_placement = True
				self.sess = tf.Session(graph=self.graph, config=config_)

	def tensorboard_init(self):
		with self.graph.as_default():
			self.saver = tf.train.Saver()
			self.merged = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, self.folder, self.runID, 'train'), self.graph)
			self.valid_writer = tf.summary.FileWriter(os.path.join(config.log_dir, self.folder, self.runID, 'valid'))

			# Save arguments
			with open(os.path.join(config.log_dir, self.folder, self.runID, 'params'), 'w') as f:
				json.dump(self.args, f)

	def save(self, step):
		with self.graph.as_default():
			path = os.path.join(config.log_dir, self.folder, self.runID,"model.ckpt")
			self.saver.save(self.sess, path, step)
			return path

	def restore_model(self, path):
		with self.graph.as_default():
			temp_saver = tf.train.Saver()
			temp_saver.restore(self.sess, tf.train.latest_checkpoint(path))

	def restore_last_checkpoint(self):
		self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(config.log_dir, self.folder ,self.runID)))

	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())

	def non_initialized_variables(self):
		with self.graph.as_default():
			global_vars = tf.global_variables()
			is_not_initialized = self.sess.run([~(tf.is_variable_initialized(var)) \
										   for var in global_vars])
			not_initialized_vars = list(compress(global_vars, is_not_initialized))
			print 'not init: ', [v.name for v in not_initialized_vars]
			if len(not_initialized_vars):
				init = tf.variables_initializer(not_initialized_vars)
				return init

	def initialize_non_init(self):
		with self.graph.as_default():
			self.sess.run(self.non_initialized_variables())

	def add_enhance_layer(self):
		with self.graph.as_default():
			self.separate
			self.enhance
			self.enhance_cost
			self.cost = self.enhance_cost
			self.freeze_all_except('enhance')
			self.optimize

	@scope
	def normalization01(self):
		min_ = tf.reduce_min(self.X, axis=[1,2], keep_dims=True)
		max_ = tf.reduce_max(self.X, axis=[1,2], keep_dims=True)
		self.X = (self.X - min_) / (max_ - min_)

	@scope
	def normalization_mean_std(self):
		mean, var = tf.nn.moments(self.X, axes=[1,2], keep_dims=True)
		self.X = (self.X - mean) / var


	@scope
	def preprocessing(self):
		self.stfts = tf.contrib.signal.stft(self.x_mix, 
			frame_length=self.window_size, 
			frame_step=self.window_size-self.hop_size,
			fft_length=self.window_size)

		self.B = tf.shape(self.x_non_mix)[0]

		self.stfts_non_mix = tf.contrib.signal.stft(tf.reshape(self.x_non_mix, [self.B*self.S, -1]), 
			frame_length=self.window_size, 
			frame_step=self.window_size-self.hop_size,
			fft_length=self.window_size)

		self.X = tf.sqrt(tf.abs(self.stfts))
		self.X_non_mix = tf.sqrt(tf.abs(self.stfts_non_mix))
		self.X_non_mix = tf.reshape(self.X_non_mix, [self.B, self.S, -1, self.F])
		self.X_non_mix = tf.transpose(self.X_non_mix, [0, 2, 3, 1])

		argmax = tf.argmax(tf.abs(self.X_non_mix), axis=3)
		self.Y = tf.one_hot(argmax, 2, 1.0, 0.0)


	@scope
	def prediction(self):
		# DPCL network

		shape = tf.shape(self.X)

		layers = [
			BLSTM(self.layer_size, 'BLSTM_1'),
			BLSTM(self.layer_size, 'BLSTM_2'),
			Conv1D([1, self.layer_size, self.embedding_size*self.F]),
			Reshape([self.B, shape[1], self.F, self.embedding_size]),
			Normalize(3)
		]

		y = f_props(layers, self.X)
		
		return y

	@scope
	def cost(self):
		# Definition of cost for DAS model

		# Get the shape of the input
		shape = tf.shape(self.Y)
		B = shape[0]
		T = shape[1]
		F = shape[2]
		S = shape[3]

		# Reshape the targets to be of shape (batch, T*F, c) and the vectors to
		# have shape (batch, T*F, K)
		Y = tf.reshape(self.Y, [B, T*F, S])
		V = tf.reshape(self.prediction, [B, T*F, self.embedding_size])

		# Compute the partition size vectors
		ones = tf.ones([B, T*F, 1])
		mul_ones = tf.matmul(tf.transpose(Y, perm=[0,2,1]), ones)
		diagonal = tf.matmul(Y, mul_ones)
		D = 1/tf.sqrt(diagonal)
		D = tf.reshape(D, [B, T*F])

		# Compute the matrix products needed for the cost function.  Reshapes
		# are to allow the diagonal to be multiplied across the correct
		# dimensions without explicitly constructing the full diagonal matrix.
		DV  = D * tf.transpose(V, perm=[2,0,1])
		DV = tf.transpose(DV, perm=[1,2,0])
		VTV = tf.matmul(tf.transpose(V, perm=[0,2,1]), DV)

		DY = D * tf.transpose(Y, perm=[2,0,1])
		DY = tf.transpose(DY, perm=[1,2,0])
		VTY = tf.matmul(tf.transpose(V, perm=[0,2,1]), DY)

		YTY = tf.matmul(tf.transpose(Y, perm=[0,2,1]), DY)

		# Compute the cost by taking the Frobenius norm for each matrix
		cost = tf.norm(VTV, axis=[-2,-1]) -2*tf.norm(VTY, axis=[-2,-1]) + tf.norm(YTY, axis=[-2,-1])

		cost = tf.reduce_mean(cost)

		tf.summary.scalar('cost', cost)
		tf.summary.scalar('1', tf.reduce_mean(tf.norm(VTV, axis=[-2,-1])))
		tf.summary.scalar('2', tf.reduce_mean(-2*tf.norm(VTY, axis=[-2,-1])))
		tf.summary.scalar('3', tf.reduce_mean(tf.norm(YTY, axis=[-2,-1])))

		return cost

	@scope
	def separate(self):
		# Input for KMeans algorithm [B, TF, E]
		input_kmeans = tf.reshape(self.prediction, [self.B, -1, self.embedding_size])
		# S speakers to separate, give self.X in input not to consider silent bins
		kmeans = KMeans(nb_clusters=self.S, nb_iterations=10, input_tensor=input_kmeans, latent_space_tensor=self.X)
		
		# Extract labels of each bins TF_i - labels [B, TF, 1]
		_ , labels = kmeans.network
		self.masks = tf.one_hot(labels, 2, 1.0, 0.0) # Create masks [B, TF, S]

		separated = tf.reshape(self.X, [self.B, -1, 1])* self.masks # [B ,TF, S] 
		separated = tf.reshape(separated, [self.B, -1, self.F, self.S])
		separated = tf.transpose(separated, [0,3,1,2]) # [B, S, T, F]
		separated = tf.reshape(separated, [self.B*self.S, -1, self.F, 1]) # [BS, T, F, 1]

		return separated

	@scope
	def enhance(self):
		# [B, S, T, F]
		separated = tf.reshape(self.separate, [self.B, self.S, -1, self.F])

		# X [B, T, F]
		# Tiling the input S time - like [ a, b, c] -> [ a, a, b, b, c, c], not [a, b, c, a, b, c]
		X_in = tf.expand_dims(self.X, 1)
		X_in = tf.tile(X_in, [1, self.S, 1, 1])
		X_in = tf.reshape(X_in, [self.B, self.S, -1, self.F])

		# Concat the binary separated input and the actual tiled input
		sep_and_in = tf.concat([separated, X_in], axis = 3)
		sep_and_in = tf.reshape(sep_and_in, [self.B*self.S, -1, 2*self.F])
		
		layers = [
			BLSTM(self.layer_size, 'BLSTM_1'),
			BLSTM(self.layer_size, 'BLSTM_2'),
			BLSTM(self.layer_size, 'BLSTM_3'),
		]

		mean, var = tf.nn.moments(sep_and_in, [1,2], keep_dims=True)
		sep_and_in = (sep_and_in - mean)/var

		y = f_props(layers, sep_and_in)
		y = tf.layers.dense(y, self.F)

		y = tf.reshape(y, [self.B, self.S, -1]) # [B, S, TF]

		y = tf.transpose(y, [0, 2, 1]) # [B, TF, S]

		y = tf.nn.softmax(y) * tf.reshape(self.X, [self.B, -1, 1]) # Apply enhanced filters # [B, TF, S] -> [BS, T, F, 1]
		# y = y * tf.reshape(self.X, [self.B, -1, 1]) # Apply enhanced filters # [B, TF, S] -> [BS, T, F, 1]
		self.cost_in = y
		y =  tf.transpose(y, [0, 2, 1])
		return tf.reshape(y , [self.B*self.S, -1, self.F, 1])

	@scope
	def enhance_cost(self):
		# Compute all permutations among the enhanced filters [B, TF, S] -> [B, TF, P, S]
		perms = list(permutations(range(self.S))) # ex with 3: [0, 1, 2], [0, 2 ,1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]
		length_perm = len(perms)

		# enhance [ B, TF, S] , X [B, T, F] -> [ B, TF, S]
		test_enhance = tf.tile(tf.reshape(tf.transpose(self.cost_in, [0,2,1]), [self.B, 1, self.S, -1]), [1, length_perm, 1, 1]) # [B, S, TF]

		
		perms = tf.reshape(tf.constant(perms), [1, length_perm, self.S, 1])
		perms = tf.tile(perms, [self.B, 1, 1, 1])

		batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1, 1]), [1, length_perm, self.S, 1])
		perm_range = tf.tile(tf.reshape(tf.range(length_perm, dtype=tf.int32), shape=[1, length_perm, 1, 1]), [self.B, 1, self.S, 1])
		indicies = tf.concat([batch_range, perm_range, perms], axis=3)

		# [B, P, S, TF]
		permuted_approx= tf.gather_nd(test_enhance, indicies)

		# X_non_mix [B, T, F, S]
		X_non_mix = tf.transpose(tf.reshape(self.X_non_mix, [self.B, 1, -1, self.S]), [0, 1, 3, 2])
		cost = tf.reduce_sum(tf.square(X_non_mix-permuted_approx), axis=-1) # Square difference on each bin 
		cost = tf.reduce_sum(cost, axis=-1) # Sum among all speakers

		cost = tf.reduce_min(cost, axis=-1) # Take the minimum permutation error

		# training_vars = tf.trainable_variables()
		# reg = []
		# for var in training_vars:
		# 	if 'enhance' in var.name:
		# 		reg.append(tf.nn.l2_loss(var))
		# reg = sum(reg)

		cost = tf.reduce_mean(cost) #+ self.adapt_front.l * reg

		# tf.summary.scalar('regularization',  reg)
		tf.summary.scalar('cost', cost)

		return cost

	@scope
	def optimize(self):
		if hasattr(self, 'trainable_variables') == False:
			self.trainable_variables = tf.global_variables()
			print 'ALL VARIABLE TRAINED'	
		print self.trainable_variables

		optimizer = AMSGrad(self.learning_rate, epsilon=0.001)
		gradients, variables = zip(*optimizer.compute_gradients(self.cost, var_list=self.trainable_variables))
		optimize = optimizer.apply_gradients(zip(gradients, variables))
		return optimize

	def train(self, X_mix, X_non_mix, step):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.x_mix: X_mix, self.x_non_mix:X_non_mix})
		# cost = self.sess.run(self.stfts, {self.x_mix: X_mix, self.x_non_mix:X_non_mix, self.I:I})
		self.train_writer.add_summary(summary, step)
		return cost

	def valid_batch(self, X_mix_valid, X_non_mix_valid):
		cost = self.sess.run(self.cost, {self.x_non_mix:X_non_mix_valid, self.x_mix:X_mix_valid})
		return cost

	def add_valid_summary(self, val, step):
		summary = tf.Summary()
		summary.value.add(tag="Valid Cost", simple_value=val)
		self.valid_writer.add_summary(summary, step)

	@staticmethod
	def load(path, modified_args):
		# Load parameters used for the desired model to load
		params_path = os.path.join(path, 'params')
		with open(params_path) as f:
			args = json.load(f)
		# Update with new args such as 'pretraining' or 'type'
		args.update(modified_args)

		# Create a new Adapt model with these parameters
		return DPCL(**args)

	def freeze_all_except(self, prefix):
		training_var = tf.trainable_variables()
		to_train = []
		for var in training_var:
			if prefix in var.name:
				to_train.append(var)
		self.trainable_variables = to_train