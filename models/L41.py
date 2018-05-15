# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.ops import BLSTM, Conv1D, Reshape, Normalize, f_props, scope, variable_summaries
from models.network import Separator

class L41Model(Separator):

	def __init__(self, graph=None, **kwargs):
		kwargs['mask_a'] = 1.0
		kwargs['mask_b'] = -1.0

		super(L41Model, self).__init__(graph, **kwargs)

		with self.graph.as_default():
			# Define the speaker vectors to use during training
			self.speaker_vectors =tf.Variable(tf.truncated_normal(
								   [self.num_speakers, self.embedding_size],
								   stddev=tf.sqrt(2/float(self.embedding_size))), name='speaker_centroids')
		self.init_separator()

	@scope
	def prediction(self):
		# L41 network
		shape = tf.shape(self.X)

		self.true_masks = 1.0 + self.y

		X_in = tf.identity(self.X)
		

		layers = [BLSTM(self.layer_size, 'BLSTM_'+str(i)) for i in range(self.nb_layers)]

		layers_sp = [
			Conv1D([1, self.layer_size, self.embedding_size*self.F]),
			Reshape([self.B, shape[1], self.F, self.embedding_size])
		]

		if self.normalize:
			layers_sp += [Normalize(3)]

		layers += layers_sp

		y = f_props(layers, X_in)
		
		return y

	@scope
	def cost(self):
		"""
		Construct the cost function op for the negative sampling cost
		"""

		# Get the embedded T-F vectors from the network
		embedding = self.prediction

	

		# Normalize the speaker vectors and collect the speaker vectors
		# corresponding to the speakers in batch
		if self.normalize:
			speaker_vectors = tf.nn.l2_normalize(self.speaker_vectors, 1)
		else:
			speaker_vectors = self.speaker_vectors

		if self.sampling is None:
			I = tf.expand_dims(self.I, axis=2) # [B, S, 1]
			# Gathering the speaker_vectors [|S|, E]
			Vspeakers = tf.gather_nd(speaker_vectors, I) # [B, S, E]
		else:
			# Get index of dominant speaker
			dominant = tf.argmax(self.y, -1) # [B, T, F]

			# For each speaker vector get the K-neighbors
			with tf.name_scope('K-Neighbors'):
				I = tf.expand_dims(self.I, axis=2)
				 
				# Gathering the speaker_vectors [B, S, E]
				Vspeakers = tf.gather_nd(speaker_vectors, I)
				
				# [B, S, 1, E]
				Vspeakers_ext = tf.expand_dims(Vspeakers, 2)

				# [1, 1, |S|, E]
				speaker_vectors_ext = tf.expand_dims(tf.expand_dims(speaker_vectors, 0), 0)

				# dot product # [B, S, |S|]
				prod_dot = tf.reduce_sum(Vspeakers_ext * speaker_vectors_ext, 3)
				
				# K neighbors [B, S, K]
				_, k_neighbors = tf.nn.top_k(prod_dot, k=self.sampling, sorted=False)

				k_neighbors = tf.reshape(k_neighbors, [-1, 1])

				# K neighbors vectors [B, S, K, E]
				k_neighbors_vectors = tf.gather_nd(speaker_vectors, k_neighbors)
				k_neighbors_vectors = tf.reshape(k_neighbors_vectors, [self.B, self.S, self.sampling, self.embedding_size])

				# [B, TF]
				dominant = tf.reshape(dominant, [self.B, -1, 1])
				
				batch_range = tf.tile(tf.reshape(tf.range(tf.cast(self.B, tf.int64), dtype=tf.int64), shape=[self.B, 1, 1]), [1, tf.shape(dominant)[1], 1])
				indices = tf.concat([batch_range, dominant], axis = 2)

				# Gathered K-nearest neighbors on each tf bins for the dominant
				# [B, T, F, K, E]
				k_neighbors_vectors_tf = tf.reshape(tf.gather_nd(k_neighbors_vectors, indices)
								,[self.B, -1, self.F, self.sampling, self.embedding_size])
				

				# []
			dominant_speaker = tf.gather(self.I, dominant) # [B, TF]
			dominant_speaker_vector = tf.gather_nd(tf.expand_dims(speaker_vectors, 1), dominant_speaker) # [B, TF, E]
			dominant_speaker_vector = tf.reshape(dominant_speaker_vector, [self.B, -1, self.F, self.embedding_size])
			dominant_speaker_vector = tf.expand_dims(dominant_speaker_vector, 3) # [B, T, F, 1, E]

			# Additional term for the loss
			doto = tf.reduce_sum(k_neighbors_vectors_tf * dominant_speaker_vector, -1)
			c = -tf.log(tf.nn.sigmoid(tf.negative(doto))) # [B, T, F, K]
			neg_sampl = tf.reduce_mean(c, -1) # [B, T, F]

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
		cost = tf.reduce_mean(cost, 3)

		if self.sampling is not None:
			cost += self.ns_rate * neg_sampl

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