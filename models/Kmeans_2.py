# -*- coding: utf-8 -*-
# My Model 
from utils.ops import ops
import tensorflow as tf
import numpy as np
from  sklearn.datasets import make_blobs

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class KMeans:

	def __init__(self, nb_clusters, nb_iterations=50, graph=None, input_tensor=None):

		self.nb_clusters = nb_clusters
		self.nb_iterations = nb_iterations

		if input_tensor == None:
			self.graph = tf.Graph()
		else:
			self.graph = tf.get_default_graph()

		with self.graph.as_default():
			if input_tensor == None:
				# Spectrogram, embeddings
				# shape = [batch, T*F , E ]
				self.X = tf.placeholder("float", [None, None, None])
			else:
				self.X = input_tensor

			self.input_dim = tf.shape(self.X)[1]
			self.B = tf.shape(self.X)[0]

			# Take randomly 'nb_clusters' vectors from X
			batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1]), [1, self.nb_clusters, 1])
			random = tf.random_uniform([self.B, self.nb_clusters, 1], minval = 0, maxval = self.input_dim - 1, dtype = tf.int32)
			indices = tf.concat([batch_range, random], axis = 2)
			self.centroids = tf.gather_nd(self.X, indices)
			self.network

		if graph == None and input_tensor == None:
			# Create a session for this model based on the constructed graph
			self.sess = tf.Session(graph = self.graph)


	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	@ops.scope
	def network(self):

		i = tf.constant(0)
		cond = lambda i, m: tf.less(i, self.nb_iterations)
		_ , self.centroids = tf.while_loop(cond, self.body,[i, self.centroids], shape_invariants=[i.get_shape(), tf.TensorShape([None, None, None])])

		return self.centroids, self.get_labels(self.centroids, self.X)


	def body(self ,i, centroids):
		with tf.name_scope('iteration'):
				# Checking the closest clusters
				labels = self.get_labels(centroids, self.X)

				# Creating the matrix equality [ B , S , TF], equality[: , s, :] = [labels == s](float32)
				cluster_range = tf.range(0, tf.shape(centroids)[1])
				equality = tf.map_fn(lambda r: tf.cast(tf.equal(labels, r), tf.float32), cluster_range, dtype=tf.float32)
				equality = tf.transpose(equality, [1 , 0, 2])

				new_centroids = tf.matmul(equality, self.X)/tf.reduce_sum(equality, axis=2, keep_dims=True)
				return [i+1, new_centroids]


	def get_labels(self, centroids, X):
		centroids_ = tf.expand_dims(centroids, 1)
		X_ = tf.expand_dims(X, 2)
		return tf.argmin(tf.norm(X_ - centroids_, axis=3), axis=2, output_type=tf.int32)


	def fit(self, X_train):
		return self.sess.run(self.network, {self.X: X_train})

if __name__ == "__main__":
	nb_samples = 10000
	E = 2
	nb_clusters = 2
	# X1 = np.random.random_sample((nb_samples/2, E))
	# X2 = np.random.random_sample((nb_samples/2, E)) + 2
	# X = np.reshape(np.concatenate((X1,X2), axis=0), (1, nb_samples ,E))
	# X = np.reshape(np.concatenate((X, X), axis=0), (2, nb_samples ,E))
	# print X.shape 

	X, y = make_blobs(n_samples=nb_samples, centers=nb_clusters, n_features=E)
	X = X[np.newaxis,:]
	y = y[np.newaxis,:]
	print y
	kmean = KMeans(nb_clusters)
	kmean.init()
	centroids, labels = kmean.fit(X)
	print labels
	print y
	if np.all((y-labels) == 0) or np.all((y+labels) == 1):
		print 'OK'

