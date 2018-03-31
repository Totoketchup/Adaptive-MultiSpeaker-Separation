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

		if self.normalize_input:
			mean, var = tf.nn.moments(self.X, axes=[1,2], keep_dims=True)
			self.X = tf.realdiv(tf.subtract(self.X, mean), tf.sqrt(var))
			self.var = var

		layers = [BLSTM(self.layer_size, 'BLSTM_'+str(i)) for i in range(self.nb_layers)]

		layers_sp = [
			Conv1D([1, self.layer_size, self.embedding_size*self.F]),
			Reshape([self.B, shape[1], self.F, self.embedding_size])
		]

		if self.normalize:
			layers_sp += [Normalize(3)]

		layers += layers_sp

		y = f_props(layers, self.X)
		
		return y

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