# -*- coding: utf-8 -*-
# My Model 
from utils.ops import ops
import tensorflow as tf
import numpy as np
from  sklearn.datasets import make_blobs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class KMeans:

	def __init__(self, nb_clusters, centroids_init=None, nb_iterations=50, graph=None, input_tensor=None):

		self.nb_clusters = nb_clusters
		self.nb_iterations = nb_iterations

		if input_tensor == None:
			self.graph = tf.Graph()
		else:
			self.graph = tf.get_default_graph()

		with self.graph.as_default():
			if input_tensor == None:
				# Spectrogram, embeddings
				# shape = [batch, L , E ]
				self.X = tf.placeholder("float", [None, None, None])
			else:
				self.X = input_tensor

			self.B = tf.shape(self.X)[0]
			self.L = tf.shape(self.X)[1]
			self.E = tf.shape(self.X)[2]

			if centroids_init is None:
				# Take randomly 'nb_clusters' vectors from X
				batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1]), [1, self.nb_clusters, 1])
				random = tf.random_uniform([self.B, self.nb_clusters, 1], minval = 0, maxval = self.L - 1, dtype = tf.int32)
				indices = tf.concat([batch_range, random], axis = 2)
				self.centroids = tf.gather_nd(self.X, indices)
			else:
				self.centroids = tf.identity(centroids_init)

			self.reshaped_X = tf.reshape(self.X, [self.B*self.L, self.E])
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
				# [B, L]
				labels = self.get_labels(centroids, self.X)

				elems_tot = (self.X, labels)
				elems_count = (tf.ones_like(self.X), labels)

				total = tf.map_fn(lambda x: tf.unsorted_segment_sum(x[0], x[1], self.nb_clusters), elems_tot, dtype=tf.float32)
				count = tf.map_fn(lambda x: tf.unsorted_segment_sum(x[0], x[1], self.nb_clusters), elems_count, dtype=tf.float32)				
				
				new_centroids = total/count	
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

