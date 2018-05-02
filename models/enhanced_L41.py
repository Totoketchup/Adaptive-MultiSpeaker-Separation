# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.ops import BLSTM, scope, variable_summaries, Conv1D
from models.network import Separator

class MyModel(Separator):

	def __init__(self, graph=None, **kwargs):
		kwargs['mask_a'] = 1.0
		kwargs['mask_b'] = -1.0

		super(MyModel, self).__init__(graph, **kwargs)

		with self.graph.as_default():
			# Define the speaker vectors to use during training
			self.speaker_vectors =tf.Variable(tf.truncated_normal(
								   [self.num_speakers, self.embedding_size],
								   stddev=tf.sqrt(2/float(self.embedding_size))), name='speaker_centroids')

		self.init_separator()

	@scope
	def prediction(self):
		# L41 network
		self.true_masks = 1.0 + self.y

		X_in = tf.identity(self.X)
		if self.abs_input:
			X_in = tf.abs(X_in)

		if self.normalize_input == '01':
			self.min_ = tf.reduce_min(X_in, axis=[1,2], keep_dims=True)
			self.max_ = tf.reduce_max(X_in, axis=[1,2], keep_dims=True)
			X_in = (X_in- self.min_) / (self.max_ - self.min_)
		elif self.normalize_input == 'meanstd':
			mean, var = tf.nn.moments(X_in, axes=[1,2], keep_dims=True)
			X_in = (X_in - mean) / tf.sqrt(var)

		f = 48
		X_in = tf.expand_dims(X_in, 3)
		X_in = tf.reshape(X_in, [-1, 80, 256, 1])
		y = tf.contrib.layers.conv2d(X_in, f, [1, 7], rate=[1,1])
		y = tf.contrib.layers.conv2d(y, f, [7, 1], rate=[1,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[4,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[8,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[16,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[32,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[1,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[2,2])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[4,4])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[8,8])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[16,16])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[32,32])
		y = tf.contrib.layers.conv2d(y, 8, [5, 5], rate=[1,1])

		y = tf.reshape(y, [self.B, 80, 256*8])

		y = BLSTM(400, 'BLSTM_1').f_prop(y)
		y = Conv1D([1, 400, self.embedding_size*self.F]).f_prop(y)
		y = tf.reshape(y, [self.B, 80, self.F, self.embedding_size])

		if self.normalize:
			y += tf.nn.l2_normalize(y)

		return y

	@scope
	def cost(self):
		"""
		Construct the cost function op for the negative sampling cost
		"""

		# Get the embedded T-F vectors from the network
		embedding = self.prediction

		# Reshape I so that it is of the correct dimension
		I = tf.expand_dims(self.I, axis=2)

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
		cost = tf.reduce_mean(cost, 3)

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