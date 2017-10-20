# My Model 
from utils.ops import ops
import config
import tensorflow as tf
import numpy as np
from  sklearn.datasets import make_blobs

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class KMeans:

	def __init__(self, nb_clusters, alpha= 5, threshold= -1):

		self.nb_clusters = nb_clusters
		self.alpha = alpha
		self.nb_iteration = 10


		self.graph = tf.Graph()

		with self.graph.as_default():
			# Spectrogram, embeddings
			# shape = [batch, T*F , E ]
			self.X = tf.placeholder("float", [1, 10000, 2])


			self.Ws = tf.cast(self.X - threshold >= 0, self.X.dtype) * self.X

			self.B = tf.shape(self.X)[0]
			self.L = tf.shape(self.X)[1]
			self.E = tf.shape(self.X)[2]

			# Take randomly 'nb_clusters' vectors from X
			batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1]), [1, self.nb_clusters, 1])
			random = tf.random_uniform([self.B, self.nb_clusters, 1], minval = 0, maxval = self.L - 1, dtype = tf.int32)
			indices = tf.concat([batch_range, random], axis = 2)
			self.centroids = tf.gather_nd(self.X, indices)

			self.network

		# Create a session for this model based on the constructed graph
		self.sess = tf.Session(graph = self.graph)


	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())

	@ops.scope
	def network(self):

		i = tf.constant(0)
		cond = lambda i, m: tf.less(i, self.nb_iteration)
		_ , self.centroids = tf.while_loop(cond, self.body,[i, self.centroids], shape_invariants=[i.get_shape(), tf.TensorShape([None, None, None])])

		return self.centroids, self.get_assignements(self.centroids, self.X)


	def body(self ,i, centroids):
		with tf.name_scope('iteration'):
				# Checking the closest clusters
				assignments = self.get_assignements(centroids, self.X)
				self.test =assignments
				# Shape (B, L, S ,1)

				X_W = X*self.Ws

				ass_W = tf.matmul(tf.transpose(assignments, [0,2,1]), self.Ws)
				ass_W_X = tf.matmul(tf.transpose(assignments, [0,2,1]), X_W)

				new_centroids = ass_W_X/ass_W
				print new_centroids
				return [i+1, new_centroids]

	def get_assignements(self, centroids, X):
		centroids = tf.expand_dims(centroids, 1)
		X = tf.expand_dims(self.X, 2)
		exp_X_cent_dist = tf.exp(-5.0*tf.norm(X - centroids, axis=3))
		a = exp_X_cent_dist/tf.reduce_sum(exp_X_cent_dist, axis=2, keep_dims=True)
		print a
		return a

	def fit(self, X_train):
		c, l = self.sess.run(self.network, {self.X: X_train})
		return c, l


from sklearn.cluster import KMeans as km

if __name__ == "__main__":
	nb_samples = 10000
	E = 2
	nb_clusters = 2
	
	X, y = make_blobs(n_samples=nb_samples, centers=nb_clusters, n_features=E)

	X_ = X[np.newaxis,:]
	y = y[np.newaxis,:]
	print y
	kmean = KMeans(nb_clusters)
	kmean.init()
	centroids, labels = kmean.fit(X_)
	print centroids
	print np.sum(labels)
	print y

	kmeans = km(n_clusters=2, random_state=0).fit(X)
	print kmeans.cluster_centers_