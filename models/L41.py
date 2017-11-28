# -*- coding: utf-8 -*-
import tensorflow as tf

from utils.ops import ops
from models.Kmeans_2 import KMeans
import config
from utils.ops.ops import BLSTM, Conv1D, Reshape, Normalize, f_props
from itertools import permutations

class L41Model:
	def __init__(self, input_tensor,
		adapt_front,
		S_tot=config.dev_clean_speakers,
		E=config.embedding_size,
		layer_size=600,
		nonlinearity='logistic',
		normalize=True):
		"""
		Initializes Lab41's clustering model.  Default architecture comes from
		the parameters used by the best performing model in the paper[1].
		[1] Hershey, John., et al. "Deep Clustering: Discriminative embeddings
			for segmentation and separation." Acoustics, Speech, and Signal
			Processing (ICASSP), 2016 IEEE International Conference on. IEEE,
			2016.
		Inputs:
			F: Number of frequency bins in the input data
			num_speakers: Number of unique speakers to train on. only use in
						  training.
			layer_size: Size of BLSTM layers
			embedding_size: Dimension of embedding vector
			nonlinearity: Nonlinearity to use in BLSTM layers (default logistic)
			normalize: Do you normalize vectors coming into the final layer?
					   (default False)
		"""

		self.F = adapt_front.N
		self.num_speakers = S_tot
		self.layer_size = layer_size
		self.embedding_size = E
		self.nonlinearity = nonlinearity
		self.normalize = normalize
		self.B = adapt_front.B
		self.S = adapt_front.S
		self.adapt_front = adapt_front

		self.graph = adapt_front.graph

		with self.graph.as_default():

			self.X, self.X_non_mix = input_tensor
			print self.X
			with tf.name_scope('create_masks'):
				# # Batch of Masks (bins label)
				# # shape = [ batch size, T, F, S]
				argmax = tf.argmax(self.X_non_mix, axis=3)
				self.y = tf.one_hot(argmax, 2, 1.0, -1.0)
				self.y_test_export = tf.reshape(self.y[:, :, :, 0], [self.B, -1])

			# Speakers indices used in the mixtures
			# shape = [ batch size, #speakers]
			self.I = adapt_front.Ind

			# Define the speaker vectors to use during training
			self.speaker_vectors =tf.Variable(tf.truncated_normal(
								   [self.num_speakers, self.embedding_size],
								   stddev=tf.sqrt(2/float(self.embedding_size))), name='speaker_centroids')

	@ops.scope
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
			# BLSTM(self.layer_size, 'BLSTM_3'),
			# BLSTM(self.layer_size, 'BLSTM_4'),
			Conv1D([1, self.layer_size, self.embedding_size*self.F]),
			Reshape([shape[0], shape[1], self.F, self.embedding_size]),
			Normalize(3)
		]

		# Produce embeddings [B, T, F, E]
		y = f_props(layers, self.X)
		
		return y

	@ops.scope
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
		print labels
		self.masks = tf.one_hot(labels, 2, 1.0, 0.0) # Create masks [B, TF, S]

		separated = tf.reshape(self.X, [self.B, -1, 1])* self.masks # [B ,TF, S] 
		separated = tf.reshape(separated, [self.B, -1, self.F, self.S])
		separated = tf.transpose(separated, [0,3,1,2]) # [B, S, T, F]
		separated = tf.reshape(separated, [self.B*self.S, -1, self.F, 1]) # [BS, T, F, 1]

		return separated

	@ops.scope
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

		# y = tf.nn.softmax(y) * tf.reshape(self.X, [self.B, -1, 1]) # Apply enhanced filters # [B, TF, S] -> [BS, T, F, 1]
		y = y * tf.reshape(self.X, [self.B, -1, 1]) # Apply enhanced filters # [B, TF, S] -> [BS, T, F, 1]
		self.cost_in = y
		y =  tf.transpose(y, [0, 2, 1])
		return tf.reshape(y , [self.B*self.S, -1, self.F, 1])

	@ops.scope
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


	@ops.scope
	def cost(self):
		"""
		Construct the cost function op for the negative sampling cost
		"""

		# Get the embedded T-F vectors from the network
		embedding = self.prediction

		# Reshape I so that it is of the correct dimension
		I = tf.expand_dims( self.I, axis=2 )

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

		# training_vars = tf.trainable_variables()
		# reg = []
		# for var in training_vars:
		# 	if 'prediction' in var.name:
		# 		reg.append(tf.nn.l2_loss(var))
		# reg = sum(reg)

		# Average the cost over all T-F elements.  Here is where weighting to
		# account for gradient confidence can occur
		cost = tf.reduce_mean(cost) 

		tf.summary.scalar('cost', cost)

		#cost = cost + 0.001*self.adapt_front.l*reg

		# tf.summary.scalar('regularized', cost)


		return cost

	def get_centroids(self):
		return self.speaker_vectors.eval()