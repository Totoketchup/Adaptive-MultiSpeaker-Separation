# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.ops import BLSTM, Conv1D, Reshape, Normalize, f_props, scope, log10
from models.network import Separator

class L41ModelV2(Separator):

	def __init__(self, graph=None, **kwargs):
		kwargs['mask_a'] = 1.0
		kwargs['mask_b'] = -1.0

		super(L41ModelV2, self).__init__(graph, **kwargs)

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
		

		layers = [BLSTM(self.layer_size, name='BLSTM_'+str(i), drop_val=self.rdropout) for i in range(self.nb_layers)]

		layers_sp = [
			Conv1D([1, self.layer_size, self.embedding_size*self.F]),
			Reshape([self.B, shape[1], self.F, self.embedding_size]),
		]

		layers += layers_sp

		y = f_props(layers, X_in)
		
		return y

	@scope
	def cost(self):
		"""
		Construct the cost function op for the negative sampling cost
		"""

		if self.loss_with_silence:
			max_ = tf.reduce_max(tf.abs(self.X), [1, 2], keep_dims=True)
			log_compare = log10(tf.divide(max_, tf.abs(self.X)))
			mask = tf.cast(log_compare < self.threshold_silence_loss, tf.float32)
			tf.summary.image('separator/silence_mask', tf.expand_dims(mask,3), max_outputs=1)
			y_a_b = self.y * tf.expand_dims(mask, 3)
			y_0_1 = (self.y + 1.0)/2.0 * tf.expand_dims(mask, 3)
		else:
			y_a_b = self.y
			y_0_1 = (self.y + 1.0)/2.0 


		tf.summary.image('mask/true/1', tf.abs(tf.expand_dims(y_0_1[:,:,:,0],3)))
		tf.summary.image('mask/true/2', tf.abs(tf.expand_dims(y_0_1[:,:,:,1],3)))


		# Get the embedded T-F vectors from the network
		embedding = self.prediction # [B, T, F, E]

		embedding_broad = tf.expand_dims(embedding, 4) # [B, T, F, E, 1]
		y_broad = tf.expand_dims(y_0_1, 3) # [B, T, F, 1, S] 
		v_mean = tf.reduce_sum(embedding_broad * y_broad, [1,2]) / ( 1e-12 + tf.expand_dims(tf.reduce_sum(y_0_1, [1,2]), 1))# [B, E, S]
		
		#
		# Reconstruction loss
		#

		with tf.name_scope('reconstruction_loss'):

			v_mean_broad = tf.expand_dims(v_mean, 1) # [B, 1, E, S]
			v_mean_broad = tf.expand_dims(v_mean_broad, 1) # [B, 1, 1, E, S]

			assignments = tf.reduce_sum(v_mean_broad * embedding_broad, 3)  # [B, T, F, S]

			assignments = tf.nn.sigmoid(assignments, -1) # [B, T, F, S]

			masked_input = tf.expand_dims(self.X_input, 3) * assignments

			# X_non_mix [B, T, F, S]			
			cost_recons = tf.reduce_mean(tf.square(self.X_non_mix - masked_input), axis=[1, 2])
			cost_recons = tf.reduce_mean(cost_recons, axis=-1) # Mean among all speakers [B, S]
			cost_recons = tf.reduce_mean(cost_recons)
			tf.summary.scalar('value', cost_recons)

		#
		# Constrast loss
		#
		with tf.name_scope('source_contrastive_loss'):

			speaker_vectors = tf.nn.l2_normalize(self.speaker_vectors, 1)
			embedding = tf.nn.l2_normalize(embedding, -1)

			I = tf.expand_dims(self.I, axis=2) # [B, S, 1]
			# Gathering the speaker_vectors [|S|, E]
			Vspeakers = tf.gather_nd(speaker_vectors, I) # [B, S, E]
			
			# Expand the dimensions in preparation for broadcasting
			Vspeakers_broad = tf.expand_dims(Vspeakers, 1)
			Vspeakers_broad = tf.expand_dims(Vspeakers_broad, 1) # [B, 1, 1, S, E]
			embedding_broad = tf.expand_dims(embedding, 3)

			# Compute the dot product between the embedding vectors and speaker
			# vectors
			dot = tf.reduce_sum(Vspeakers_broad * embedding_broad, 4)

			# Compute the cost for every element

			sc_cost = -tf.log(tf.nn.sigmoid(y_a_b * dot))

			sc_cost = tf.reduce_mean(sc_cost, 3) # Average the cost over all speakers in the input
			sc_cost = tf.reduce_mean(sc_cost, 0)	# Average the cost over all batches
			sc_cost = tf.reduce_mean(sc_cost) 
			tf.summary.scalar('value', sc_cost)

		cost = sc_cost + cost_recons
		tf.summary.scalar('total', cost)

		return cost