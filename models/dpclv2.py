# My Model 
from utils.ops import ops
from utils.ops.ops import BLSTM, Conv1D, Reshape, Normalize, f_props, Dropout
from tensorflow.contrib.tensorboard.plugins import projector
from models.Kmeans_2 import KMeans

# from utils.postprocessing.reconstruction import 

import os
import config
import tensorflow as tf

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class DPCL:
	def __init__(self, input_tensor, adapt, E=config.embedding_size, threshold=config.threshold):

		self.B = adapt.B
		self.S = adapt.S
		self.F = adapt.N    # Freqs size
		self.E = E    # Embedding size
		self.threshold = threshold # Threshold for silent weights
		self.graph = adapt.graph
		self.training = adapt.training


		self.X = input_tensor[0]
		self.X_non_mix = input_tensor[1]
		with tf.name_scope('create_masks'):
			self.Y = tf.cond(adapt.training, 
				lambda: tf.one_hot(tf.argmax(self.X_non_mix, axis=3), 2, 1.0, 0.0), 
				lambda: tf.constant(1.0))

	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	@ops.scope
	def prediction(self):
		# DAS network

		shape = tf.shape(self.X)

		layers = [
		BLSTM(600, 'BLSTM_1', dropout=True),
		BLSTM(600, 'BLSTM_2', dropout=True),
		BLSTM(600, 'BLSTM_3', dropout=True),
		BLSTM(600, 'BLSTM_4', dropout=True),
		Dropout(0.5, train=self.training),
		Conv1D([1, 600, self.E*self.F]),
		Reshape([self.B, shape[1], self.F, self.E]),
		Normalize(3)
		]

		y = f_props(layers, self.X)
		
		# self.embedding.tensor_name = y.name

		return y
	@ops.scope
	def enhancement(self):
		# DAS network

		shape = tf.shape(self.prediction)

		layers = [
		BLSTM(600, 'BLSTM_1'),
		BLSTM(600, 'BLSTM_2'),
		Conv1D([1, 600, self.E*self.F]),
		Reshape([self.B, shape[1], self.F, self.E]),
		Normalize(3)
		]

		y = f_props(layers, self.X)
		
		# self.embedding.tensor_name = y.name

		return y
	@ops.scope
	def cost(self):
		# Definition of cost for DAS model

		# Get the shape of the input
		shape = tf.shape(self.Y)
		print self.prediction
		# Reshape the targets to be of shape (batch, T*F, c) and the vectors to
		# have shape (batch, T*F, K)
		Y = tf.reshape(self.Y, [shape[0], shape[1]*shape[2], shape[3]])
		V = tf.reshape(self.prediction, [shape[0], shape[1]*shape[2], self.E])

		# Compute the partition size vectors
		ones = tf.ones([shape[0], shape[1]*shape[2], 1])
		mul_ones = tf.matmul(tf.transpose(Y, perm=[0,2,1]), ones)
		diagonal = tf.matmul(Y, mul_ones)
		D = 1/tf.sqrt(diagonal)
		D = tf.reshape(D, [shape[0], shape[1]*shape[2]])

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

	@ops.scope
	def separate(self):
		input_kmeans = tf.reshape(self.prediction, [self.B, -1, self.E])
		kmeans = KMeans(nb_clusters=2, nb_iterations=50, input_tensor=input_kmeans)
		_ , labels = kmeans.network
		masks = tf.one_hot(labels, 2, 1.0, 0.0)
		separated = tf.reshape(self.X, [self.B, -1, 1])* masks # [B ,TF, S] 
		separated = tf.reshape(separated, [self.B, -1, self.F, self.S])
		separated = tf.transpose(separated, [0,3,1,2])
		separated = tf.reshape(separated, [self.B*self.S, -1, self.F, 1])
		return separated

	@ops.scope
	def optimize(self):
		return tf.train.AdamOptimizer(0).minimize(self.cost)

	def train(self, X_train, Y_train, x, step):
		cost, _, summary = self.sess.run([self.cost, self.optimize, self.merged],
			{self.X: X_train, self.Y: Y_train, self.X_raw: x})
		self.train_writer.add_summary(summary, step)
		return cost


	def save(self, step):
		self.saver.save(self.sess, os.path.join('log/', "deep_clustering_model.ckpt"), step)

	def embeddings(self, X):
		V = self.sess.run(self.prediction, {self.X: X})
		return V


