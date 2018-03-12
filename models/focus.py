# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.ops import BLSTM, Conv1D, Reshape, Normalize, f_props, scope
from models.network import Separator
import numpy as np

class FocusModel(Separator):

	def __init__(self, graph=None, **kwargs):
		kwargs['mask_a'] = 1.0
		kwargs['mask_b'] = 0.0

		super(FocusModel, self).__init__(graph, **kwargs)
		self.focus_dim = 2

		with self.graph.as_default():
			# Define the speaker vectors to use during training
			self.speaker_focus_vector =tf.Variable(tf.truncated_normal(
								   [self.num_speakers, self.focus_dim],
								   stddev=tf.sqrt(2/float(self.focus_dim))), name='speaker_centroids')

			# self.y [B, T, F, S]
			y = []
			masks = np.eye(self.S, dtype=bool)
			for i in range(self.S):
				mask = masks[i]
				mask_not = np.invert(mask)

				y_i_tiled = tf.transpose(tf.tile(tf.expand_dims(self.y[:,:,:,i], 3), [1,1,1,self.S]), [3,0,1,2])
				neg_y_i_tiled = 1.0 - y_i_tiled

				y1 = tf.boolean_mask(y_i_tiled, mask)
				y2 = tf.boolean_mask(neg_y_i_tiled, mask_not)
				
				y.append(tf.transpose(y1+y2, [1,2,3,0]))
			
			self.y = tf.stack(y)

		self.init_separator()

	@scope
	def prediction(self):
		# Focus network

		shape = tf.shape(self.X)

		channels = []

		for i in range(self.S):
			V = tf.gather(self.speaker_focus_vector, self.I[:, i]) # [B, focus_dim]

			V = tf.tile(tf.reshape(V, [-1, 1, self.focus_dim]), [1, tf.shape(self.X)[-2], 1])

			layers = [BLSTM(self.layer_size, 'BLSTM_'+str(i)+'_'+str(j)) for j in range(self.nb_layers)]

			layers_sp = [
				Conv1D([1, self.layer_size, self.embedding_size*self.F]),
				Reshape([self.B, shape[1], self.F, self.embedding_size]),
				Normalize(3)
			]

			layers += layers_sp
			input_ = tf.concat([self.X, V], -1)

			channels.append(f_props(layers, input_))
		
		return channels

	@scope
	def cost(self):
		"""
		Construct the cost function op for the negative sampling cost
		"""

		# Get the embedded T-F vectors from the network
		channels = self.prediction

		for i, embedding in enumerate(channels):

			# Definition of cost for DPCL model

			# Get the shape of the input
			y = self.y[i]

			shape = tf.shape(y)
			B = shape[0]
			T = shape[1]
			F = shape[2]
			S = shape[3]

			# Reshape the targets to be of shape (batch, T*F, c) and the vectors to
			# have shape (batch, T*F, K)
			Y = tf.reshape(y, [B, T*F, S])
			V = tf.reshape(embedding, [B, T*F, self.embedding_size])

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

		return cost