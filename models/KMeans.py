# My Model 
from utils.ops import ops

import os
import config
import tensorflow as tf
import numpy as np

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class KMeans:

	def __init__(self, nb_clusters, T, batch_size= config.batch_size, alpha= 5, F=config.fftsize//2, E=config.embedding_size, threshold= config.threshold):

		self.nb_clusters = nb_clusters
		self.T = T
		self.F = F
		self.E = E
		self.alpha = alpha
		self.nb_iteration = 10
		self.batch_size = batch_size

		self.graph = tf.Graph()

		with self.graph.as_default():
			# Spectrogram, embeddings
			# shape = [batch, T*F , E ]
			self.X = tf.placeholder("float", [batch_size, T*F, E])

			self.Ws = tf.cast(self.X - threshold > 0, self.X.dtype) * self.X

			# Centroids used for each cluster
			# shape = [ batch, T*F, embedding size]
			self.centroids = tf.Variable(
				tf.truncated_normal([batch_size, nb_clusters, E], 
				stddev=tf.sqrt(2/float(E))),
				name='centroids')

			self.labels_ = tf.zeros([batch_size, T*F], dtype=tf.int32, name='labels')

			self.assignments = tf.zeros([batch_size, T*F, nb_clusters])

			self.network

			self.saver = tf.train.Saver()

		# Create a session for this model based on the constructed graph
		self.sess = tf.Session(graph = self.graph)


	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	@ops.scope
	def network(self):

		# Shape (B, TF, 1, 1)
		Ws = tf.expand_dims(self.Ws, 2)

		# Shape (B, TF, 1, E)
		X = tf.expand_dims(self.X, 2)

		for i in range(self.nb_iteration):
			# Shape (B, TF, M ,1)
			self.assignments = tf.expand_dims(self.assignments, 3)
			# print'Assignment ', self.assignments.shape
			# Shape (B, 1, M, E)
			self.centroids = tf.expand_dims(self.centroids, 1)
			# print ' centroids ', self.centroids.shape
			diff_norm = -self.alpha*tf.norm((X - self.centroids), axis=3)

			self.assignments = tf.exp(diff_norm)/(tf.expand_dims(tf.reduce_sum(tf.exp(diff_norm), axis=2),axis=2))
			# print 'Assignment ', self.assignments.shape
			self.assignments = tf.expand_dims(self.assignments, 3)
			# print 'Assignment ', self.assignments.shape
			# print 'Centroids ', self.centroids.shape
			# print 'X ', self.X.shape
			# print 'Ws ', self.Ws.shape

			self.centroids = tf.reduce_sum(self.assignments*Ws*X/self.assignments*Ws, axis=1)

		return tf.reshape(tf.argmax(self.assignments, axis=2), [self.batch_size, self.T, self.F])

	def fit(self, X_train):
		labels = self.sess.run(self.network, {self.X: X_train})
		return labels


if __name__ == "__main__":
	T = 40
	X = np.random.rand(32, T*config.fftsize//2, config.embedding_size)

	kmean = KMeans(3, T)
	kmean.init()

	label = kmean.fit(X)
	print label.shape