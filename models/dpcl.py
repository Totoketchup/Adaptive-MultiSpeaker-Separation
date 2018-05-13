# -*- coding: utf-8 -*-
from utils.ops import BLSTM, Conv1D, Reshape, Normalize, f_props, scope
from network import Separator
import tensorflow as tf

############################################
#       Deep Clustering Architecture       #
############################################

class DPCL(Separator):

	def __init__(self, graph=None, **kwargs):
		kwargs['mask_a'] = 1.0
		kwargs['mask_b'] = 0.0

		super(DPCL, self).__init__(graph, **kwargs)
		self.init_separator()

	@scope
	def prediction(self):
		# DPCL network
		self.true_masks = self.y

		shape = tf.shape(self.X)

		layers = [BLSTM(self.layer_size, 'BLSTM_'+str(i)) for i in range(self.nb_layers)]

		layers_sp = [
			Conv1D([1, self.layer_size, self.embedding_size*self.F]),
			Reshape([self.B, shape[1], self.F, self.embedding_size]),
			Normalize(3)
		]

		layers += layers_sp

		y = f_props(layers, self.X)
		
		return y

	@scope
	def cost(self):
		# Definition of cost for DPCL model

		# Get the shape of the input
		shape = tf.shape(self.y)
		B = shape[0]
		T = shape[1]
		F = shape[2]
		S = shape[3]

		# Reshape the targets to be of shape (batch, T*F, c) and the vectors to
		# have shape (batch, T*F, K)
		Y = tf.reshape(self.y, [B, T*F, S])
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