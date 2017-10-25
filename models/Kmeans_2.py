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

	def __init__(self, nb_clusters, centroids_init=None, nb_tries=10, nb_iterations=10, graph=None, input_tensor=None):

		self.nb_clusters = nb_clusters
		self.nb_iterations = nb_iterations
		self.nb_tries = nb_tries

		if input_tensor == None:
			self.graph = tf.Graph()
		else:
			self.graph = tf.get_default_graph()

		with self.graph.as_default():
			if input_tensor == None:
				# Spectrogram, embeddings
				# shape = [batch, L , E ]
				self.X_in = tf.placeholder("float", [None, None, None])
			else:
				self.X_in = input_tensor

			self.b = tf.shape(self.X_in)[0]
			self.X = tf.tile(self.X_in, [self.nb_tries, 1, 1])

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

		centroids = tf.expand_dims(self.centroids, 1)
		X = tf.expand_dims(self.X, 2)
		inertia = tf.reduce_sum(tf.norm(X - centroids, axis=3), axis=[1,2])

		inertia = tf.reshape(inertia, [self.b, self.nb_tries])
		bests = tf.argmin(inertia, 1, output_type=tf.int32)
		index = bests + tf.range(self.b)*self.nb_tries

		self.centroids = tf.reshape(self.centroids, [self.b*self.nb_tries, self.nb_clusters, self.E])

		self.centroids = tf.gather(self.centroids, index)
		print self.centroids
		return self.centroids, self.get_labels(self.centroids, self.X_in)


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
		return self.sess.run(self.network, {self.X_in: X_train})

from sklearn.cluster import KMeans as km

if __name__ == "__main__":
	nb_samples = 100
	E = 50
	nb_clusters = 5
	error = 0
	TOTAL = 100
	nb_err = 0.0
	kmean = KMeans(nb_clusters, nb_tries=1, nb_iterations=20)
	

	for i in range(TOTAL):	
		X, y = make_blobs(n_samples=nb_samples, centers=nb_clusters, n_features=E, cluster_std=2.0)
		X_ = X[np.newaxis,:]
		X_ = np.concatenate([X_,X_], axis=0)
		y = y[np.newaxis,:]
		
		kmean.init()
		centroids, labels = kmean.fit(X_)
		print centroids.shape
		centroids = np.reshape(centroids, (nb_clusters, E))
		kmeans = km(n_clusters=nb_clusters, random_state=0, ).fit(X)
		error = np.sum(np.square(np.sort(centroids,0) - np.sort(kmeans.cluster_centers_,0)))
		if error > 0.1:
			print error
			nb_err += 1.0

	print nb_err / float(TOTAL)

	print error
