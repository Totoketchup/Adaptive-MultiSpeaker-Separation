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

	def __init__(self, nb_clusters, centroids_init=None, nb_tries=3, nb_iterations=10, graph=None, input_tensor=None, latent_space_tensor=None, ):

		self.nb_clusters = nb_clusters
		self.nb_iterations = nb_iterations
		self.nb_tries = nb_tries
		self.latent_space_tensor = latent_space_tensor

		if input_tensor is None:
			self.graph = tf.Graph()
		else:
			self.graph = tf.get_default_graph()

		with self.graph.as_default():
			if input_tensor is None:
				# Spectrogram, embeddings
				# shape = [batch, L , E ]
				self.X_in = tf.placeholder("float", [None, None, None], name='Kmeans_input')
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
				self.centroids = tf.tile(self.centroids, [self.nb_tries, 1 , 1])

			if not self.latent_space_tensor is None:
				self.W_0_no_try = tf.reshape(tf.cast(self.latent_space_tensor > 1e-3, tf.float32), [self.b, self.L, 1])
				self.W_0 = tf.tile(self.W_0_no_try, [self.nb_tries, 1 , 1])

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
		labels = self.get_labels(self.centroids, self.X_in, self.W_0_no_try)
		return self.centroids, labels


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


	def get_labels(self, centroids, X, W_0_spe=None):
		centroids_ = tf.expand_dims(centroids, 1)
		if not self.latent_space_tensor is None:
			if W_0_spe is None:
				X = X*self.W_0
			else:
				X = X*W_0_spe
		X_ = tf.expand_dims(X, 2)
		distances = tf.norm(X_ - centroids_, axis=3)
		return tf.argmin(distances, axis=2, output_type=tf.int32)

	def fit(self, X_train):
		return self.sess.run(self.network, {self.X_in: X_train})

# from sklearn.cluster import KMeans as km

if __name__ == "__main__":
	nb_samples = 100
	E = 4
	nb_clusters = 4
	error = 0
	TOTAL = 1
	nb_err = 0.0
	kmean = KMeans(nb_clusters, nb_tries=3, nb_iterations=10)
	

	for i in range(TOTAL):	
		X, y = make_blobs(n_samples=nb_samples, centers=nb_clusters, n_features=E, cluster_std=10)
		X_ = X[np.newaxis,:]
		X_ = np.concatenate([X_, X_], axis=0)
		y = y[np.newaxis,:]
		
		kmean.init()
		centroids, labels, soft_labels = kmean.fit(X_)
		print centroids.shape
		# print labels
		print soft_labels
		# u = tf.one_hot(labels, 2, 1.0, 0.0, name='masks')
		# masked_val = X_ * u
		# t = tf.transpose(masked_val, [0,2,1])
		# mixed = tf.reshape(t , [2*nb_clusters, -1])
		# # print masked_val
		# # with tf.Session().as_default():
		# # 	print mixed.eval()

		# centroids = np.reshape(centroids, (2, nb_clusters, E))
		# kmeans = km(n_clusters=nb_clusters, random_state=0, ).fit(X)
