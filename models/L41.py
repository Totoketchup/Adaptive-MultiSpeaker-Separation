# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.tools import args_to_string
import haikunator
from models.Kmeans_2 import KMeans
from utils.ops import BLSTM, Conv1D, Reshape, Normalize, f_props, scope, AMSGrad, variable_summaries
from itertools import permutations
import os
import config

class L41Model:

	def __init__(self, adapt=None, **kwargs):

		if adapt is not None:

			self.num_speakers = kwargs['tot_speakers']
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
					self.y = tf.one_hot(argmax, 2, 1.0, -1.0)
					self.y_test_export = tf.reshape(self.y[:, :, :, 0], [self.B, -1])

				# Speakers indices used in the mixtures
				# shape = [ batch size, #speakers]
				self.I = adapt.Ind

				# Define the speaker vectors to use during training
				self.speaker_vectors =tf.Variable(tf.truncated_normal(
									   [self.num_speakers, self.embedding_size],
									   stddev=tf.sqrt(2/float(self.embedding_size))), name='speaker_centroids')

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
				self.num_speakers = kwargs['tot_speakers']
				self.layer_size = kwargs['layer_size']
				self.embedding_size = kwargs['embedding_size']
				self.nonlinearity = kwargs['nonlinearity']
				self.normalize = kwargs['normalize']
				self.learning_rate = kwargs['learning_rate']

				# Run ID for tensorboard
				self.runID = 'L41_STFT' + '-' + haikunator.Haikunator().haikunate()
				print 'ID : {}'.format(self.runID)
				if kwargs is not None:
					self.runID += args_to_string(kwargs)

				# Placeholder tensor for the mixed signals
				self.x_mix = tf.placeholder("float", [None, None])

				# Placeholder tensor for non mixed input data [B, T, F, S]
				# Place holder for non mixed signals [B, S, L]
				self.x_non_mix = tf.placeholder("float", [None, None, None])

				self.I = tf.placeholder(tf.int32, [None, None], name='indicies')

				# Define the speaker vectors to use during training
				self.speaker_vectors =tf.Variable(tf.truncated_normal(
									   [self.num_speakers, self.embedding_size],
									   stddev=tf.sqrt(2/float(self.embedding_size))), name='speaker_centroids')

				self.preprocessing
				self.normalization01
				self.prediction
				self.cost
				self.optimize

				config_ = tf.ConfigProto()
				config_.gpu_options.allow_growth = True
				config_.allow_soft_placement = True
				self.sess = tf.Session(graph=self.graph, config=config_)

	def tensorboard_init(self):
		with self.graph.as_default():
			self.merged = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, self.folder, self.runID, 'train'), self.graph)
			self.valid_writer = tf.summary.FileWriter(os.path.join(config.log_dir, self.folder, self.runID, 'valid'), self.graph)
			self.saver = tf.train.Saver()

	def save(self, step):
		path = os.path.join(config.log_dir, self.folder ,self.runID, "model.ckpt")
		self.saver.save(self.sess, path, step)
		return path

	def restore_last_checkpoint(self):
		self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(config.log_dir, self.folder ,self.runID)))


	def init(self):
			with self.graph.as_default():
				self.sess.run(tf.global_variables_initializer())
 	
	@scope
	def preprocessing(self):
		self.stfts = tf.contrib.signal.stft(self.x_mix, 
			frame_length=self.window_size, 
			frame_step=self.window_size-self.hop_size,
			fft_length=self.window_size)

		self.B = tf.shape(self.x_non_mix)[0]
		self.S = tf.shape(self.x_non_mix)[1]

		self.stfts_non_mix = tf.contrib.signal.stft(tf.reshape(self.x_non_mix, [self.B*self.S, -1]), 
			frame_length=self.window_size, 
			frame_step=self.window_size-self.hop_size,
			fft_length=self.window_size)

		self.X = tf.sqrt(tf.abs(self.stfts))
		self.X_non_mix = tf.sqrt(tf.abs(self.stfts_non_mix))
		self.X_non_mix = tf.reshape(self.X_non_mix, [self.B, self.S, -1, self.F])
		self.X_non_mix = tf.transpose(self.X_non_mix, [0, 2, 3, 1])

		argmax = tf.argmax(tf.abs(self.X_non_mix), axis=3)
		self.y = tf.one_hot(argmax, 2, 1.0, -1.0)

	@scope
	def normalization01(self):
		min_ = tf.reduce_min(self.X, axis=[1,2], keep_dims=True)
		max_ = tf.reduce_max(self.X, axis=[1,2], keep_dims=True)
		self.X_norm = (self.X - min_) / (max_ - min_)

	@scope
	def normalization_mean_std(self):
		mean, var = tf.nn.moments(self.X, axes=[1,2], keep_dims=True)
		self.X_norm = (self.X - mean) / var

	@scope
	def prediction(self):
		"""
		Construct the op for the network used in [1].  This consists of two
		BLSTM layers followed by a dense layer giving a set of T-F vectors of
		dimension embedding_size
		"""
		shape = tf.shape(self.X)	

		layers = [
			BLSTM(self.layer_size, 'BLSTM_1'),#, dropout=True, drop_val=0.9),
			BLSTM(self.layer_size, 'BLSTM_2'),#, dropout=True, drop_val=0.9),
			Conv1D([1, self.layer_size, self.embedding_size*self.F]),
			Reshape([shape[0], shape[1], self.F, self.embedding_size]),
			Normalize(3)
		]
		# Produce embeddings [B, T, F, E]
		y = f_props(layers, self.X_norm)
		
		return y

	@scope
	def separate(self):
		# TODO for only when Ind is available, speaker info is given
		# Ind [B, S, 1]
		# Ind = tf.expand_dims(self.I, 2)

		# U [S_tot, E]
		# centroids = tf.gather_nd(self.speaker_vectors, Ind)

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
			# BLSTM(self.layer_size, 'BLSTM_4')
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
	def cost(self):
		"""
		Construct the cost function op for the negative sampling cost
		"""

		# Get the embedded T-F vectors from the network
		embedding = self.prediction

		# Reshape I so that it is of the correct dimension
		I = tf.expand_dims(self.I, axis=2 )

		# Normalize the speaker vectors and collect the speaker vectors
		# corresponding to the speakers in batch
		if self.normalize:
			speaker_vectors = tf.nn.l2_normalize(self.speaker_vectors, 1)
		else:
			speaker_vectors = self.speaker_vectors
		Vspeakers = tf.gather_nd(speaker_vectors, I)

		# Expand the dimensions in preparation for broadcasting
		Vspeakers_broad = tf.expand_dims(Vspeakers, 1)
		Vspeakers_broad = tf.expand_dims(Vspeakers_broad, 1)
		embedding_broad = tf.expand_dims(embedding, 3)

		# Compute the dot product between the embedding vectors and speaker
		# vectors
		dot = tf.reduce_sum(Vspeakers_broad * embedding_broad, 4)

		# Compute the cost for every element
		cost = -tf.log(tf.nn.sigmoid(self.y * dot))

		# Average the cost over all speakers in the input
		cost = tf.reduce_sum(cost, 3)

		# Average the cost over all batches
		cost = tf.reduce_mean(cost, 0)

		training_vars = tf.trainable_variables()
		for var in training_vars:
			if 'prediction' in var.name:
				variable_summaries(var)

		# Average the cost over all T-F elements.  Here is where weighting to
		# account for gradient confidence can occur
		cost = tf.reduce_mean(cost) 

		tf.summary.scalar('cost', cost)

		#cost = cost + 0.001*self.adapt_front.l*reg

		# tf.summary.scalar('regularized', cost)


		return cost

	@scope
	def optimize(self):
		# optimizer = self.select_optimizer(self.optimizer)(self.learning_rate)
		optimizer = AMSGrad(self.learning_rate, epsilon=0.001)
		opt = optimizer.minimize(self.cost)
		return opt

	def get_centroids(self):
		return self.speaker_vectors.eval()

	def train(self, X_mix, X_non_mix, I, step):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.x_mix: X_mix, self.x_non_mix:X_non_mix, self.I:I})
		# cost = self.sess.run(self.stfts, {self.x_mix: X_mix, self.x_non_mix:X_non_mix, self.I:I})
		self.train_writer.add_summary(summary, step)
		return cost

	def valid_batch(self, X_mix_valid, X_non_mix_valid, I):
		cost = self.sess.run(self.cost, {self.x_non_mix:X_non_mix_valid, self.x_mix:X_mix_valid,self.I:I})
		return cost

	def add_valid_summary(self, val, step):
		summary = tf.Summary()
		summary.value.add(tag="Valid Cost", simple_value=val)
		self.valid_writer.add_summary(summary, step)
