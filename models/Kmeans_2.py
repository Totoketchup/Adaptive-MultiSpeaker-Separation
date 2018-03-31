# -*- coding: utf-8 -*-
# My Model 
from utils.ops import scope
import tensorflow as tf
import numpy as np
from  sklearn.datasets import make_blobs

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class KMeans:

	def __init__(self, nb_clusters, centroids_init=None, nb_tries=10, nb_iterations=10, input_tensor=None, latent_space_tensor=None, ):

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

			# mean, _ = tf.nn.moments(self.X_in, axes=-1, keep_dims=True)

			self.X_in = tf.nn.l2_normalize(self.X_in, axis=-1)

			self.b = tf.shape(self.X_in)[0]
			self.X = tf.expand_dims(self.X_in, 1)
			self.X = tf.tile(self.X, [1, self.nb_tries, 1, 1])

			self.L = tf.shape(self.X)[-2]
			self.E = tf.shape(self.X)[-1]
			self.X = tf.reshape(self.X, [-1, self.L, self.E])

			self.B = tf.shape(self.X)[0]

			self.ones = tf.ones_like(self.X, tf.float32)

			self.shifting = tf.tile(tf.expand_dims(tf.range(self.B)*self.nb_clusters, 1), [1, self.L])

			if centroids_init is None:
				def random_without_replace(b, l):
					a =  np.array([np.random.choice(range(l), size=self.nb_clusters, replace=False) for _ in range(b)])
					return a.astype(np.int32)
				
				y = tf.py_func(random_without_replace, [self.B, self.L], tf.int32)
				random = tf.reshape(y, [self.B, self.nb_clusters, 1])

				# Take randomly 'nb_clusters' vectors from X
				batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1]), [1, self.nb_clusters, 1])
				indices = tf.concat([batch_range, random], axis = 2)
				self.centroid_init = tf.gather_nd(self.X, indices)
			else:
				self.centroids = tf.identity(centroids_init)
				self.centroids = tf.tile(self.centroids, [self.nb_tries, 1 , 1])

			if not self.latent_space_tensor is None:
				self.W_0_no_try = tf.reshape(tf.cast(self.latent_space_tensor > 1e-3, tf.float32), [self.b, self.L, 1])
				self.W_0 = tf.tile(self.W_0_no_try, [self.nb_tries, 1 , 1])

			self.network

	@scope
	def network(self):

		i = tf.constant(0)
		init = [i, self.centroid_init, self.get_labels(self.centroid_init, self.X)]
		invariant = [i.get_shape(), tf.TensorShape([None, self.nb_clusters, None]), tf.TensorShape([None, None])]

		cond = lambda i, c, l: tf.less(i, self.nb_iterations)
		_ , centroids, labels = tf.while_loop(cond, self.body, init, shape_invariants=invariant)

		inertia = self.get_inertia(centroids)

		inertia = tf.reshape(inertia, [self.b, self.nb_tries])
		bests = tf.argmin(inertia, 1, output_type=tf.int32)
		index = bests + tf.range(self.b)*self.nb_tries

		centroids = tf.reshape(centroids, [self.b*self.nb_tries, self.nb_clusters, self.E])
		centroids = tf.gather(centroids, index)

		labels = tf.gather(labels, index)
		return centroids, labels


	def get_inertia(self, centroids):
		with tf.name_scope('inertia'):
			labels = self.get_labels(centroids, self.X)

			# centroids [b*nb_tries, C, E]
			centroids_flattened = tf.reshape(centroids, [self.B*self.nb_clusters, self.E])
			
			# Add + [0,1,2]
			idx_flattened = tf.reshape(labels + self.shifting, [self.B*self.L]) # [b*nb_tries*L]
			X_flattened = tf.reshape(self.X, [self.B*self.L, self.E]) # [b*nb_tries*L, E]

			dist = tf.reduce_sum(tf.square(X_flattened - tf.gather(centroids_flattened, idx_flattened)), -1)

			total = tf.unsorted_segment_sum(dist, idx_flattened, self.B*self.nb_clusters)
			count = tf.unsorted_segment_sum(tf.ones_like(dist), idx_flattened, self.B*self.nb_clusters)

			# [self.B*self.nb_clusters]
			inertia = total / count
			inertia = tf.reshape(inertia, [self.B, self.nb_clusters])
			inertia = tf.reduce_sum(inertia, -1)

			return inertia
				
	def body(self ,i, centroids, labels):
		with tf.name_scope('iteration'):
			# Add + [0,1,2]
			labels_flattened = tf.reshape(labels + self.shifting, [-1])
			X_flattened = tf.reshape(self.X, [-1, self.E])

			# total : [
			total = tf.unsorted_segment_sum(X_flattened, labels_flattened, self.B*self.nb_clusters)
			count = tf.unsorted_segment_sum(tf.ones_like(X_flattened), labels_flattened, self.B*self.nb_clusters)

			new_centroids = total/count
			new_centroids = tf.reshape(new_centroids, [self.B, self.nb_clusters, self.E])

			return [i+1, new_centroids, self.get_labels(new_centroids, self.X)]

	def get_labels(self, centroids, X, W_0_spe=None):
		if self.latent_space_tensor is not None:
			if W_0_spe is None:
				X = X*self.W_0
			else:
				X = X*W_0_spe

		X_ = tf.expand_dims(X, 2)
		centroids_ = tf.expand_dims(centroids, 1)

		distances = tf.sqrt(tf.reduce_sum(tf.square(X_ - centroids_), axis=3))
		return tf.argmin(distances, axis=2, output_type=tf.int32)

	def fit(self, X_train):
		with tf.Session(graph=self.graph) as sess:
			sess.run(tf.global_variables_initializer())
			return sess.run(self.network, {self.X_in: X_train})

# from sklearn.cluster import KMeans as km

if __name__ == "__main__":
	nb_samples = 100
	E = 10
	nb_clusters = 3
	error = 0
	TOTAL = 1
	nb_err = 0.0
	kmean = KMeans(nb_clusters, nb_tries=10, nb_iterations=10)
	

	for i in range(TOTAL):
		X, y = make_blobs(n_samples=nb_samples, centers=nb_clusters, n_features=E, cluster_std=0.01)
		X_ = X[np.newaxis,:]
		X_ = np.concatenate([X_, X_], axis=0)
		y = y[np.newaxis,:]
		
		centroids, labels = kmean.fit(X_)

		print labels
		print y

	
		# print labels
		# u = tf.one_hot(labels, 2, 1.0, 0.0, name='masks')
		# masked_val = X_ * u
		# t = tf.transpose(masked_val, [0,2,1])
		# mixed = tf.reshape(t , [2*nb_clusters, -1])
		# # print masked_val
		# # with tf.Session().as_default():
		# # 	print mixed.eval()

		# centroids = np.reshape(centroids, (2, nb_clusters, E))
		# kmeans = km(n_clusters=nb_clusters, random_state=0, ).fit(X)
