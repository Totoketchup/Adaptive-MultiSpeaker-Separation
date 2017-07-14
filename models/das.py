# My Model 
from utils.ops import ops
from utils.ops.ops import Residual_Net, Conv1D, Reshape, Dense

import config
import tensorflow as tf

#############################################
# 		Deep Adaptive Separator Model 		#
#############################################

class DAS:

	def __init__(self, S, T, fftsize=config.fftsize//2, E=config.embedding_size, threshold=config.threshold, l=0.2):

		self.F = fftsize	# Freqs size
		self.E = E 			# Embedding size
		self.S = S 			# Total number of speakers
		self.T = T          # Spectrograms length
		self.threshold = threshold # Threshold for silent weights
		self.l = l
		
		self.graph = tf.Graph()

		with self.graph.as_default():
			# Batch of spectrogram chunks - Input data
			# shape = [ batch size , chunk size, F ]
			self.X = tf.placeholder("float", [None, None, self.F])

			# Batch of Masks (bins label)
			# shape = [ batch size, chunk size, F, #speakers ]
			self.Y = tf.placeholder("float", [None, None, self.F, None])

			# Speakers indicies used in the mixtures
			# shape = [ batch size, #speakers]
			self.Ind = tf.placeholder(tf.int32, [None,None])

			# Placeholder for the 'dropout', telling if the network is 
			# currently learning or not
			self.training = tf.placeholder(tf.bool)

			self.Ws = tf.cast(self.X - threshold > 0, self.X.dtype) * self.X

			# The centroids used for each speaker
			# shape = [ #tot_speakers, embedding size]
			self.speaker_centroids = tf.Variable(tf.truncated_normal([self.S,self.E], stddev=tf.sqrt(2/float(self.E))))

			self.prediction
			self.cost
			self.optimize

			self.saver = tf.train.Saver()
			self.train_writer = tf.summary.FileWriter('log/', self.graph)
			self.merged = tf.summary.merge_all()


		# Create a session for this model based on the constructed graph
		self.sess = tf.Session(graph = self.graph)


	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	@ops.scope
	def prediction(self):
		# DAS network

		k = [1, 32, 64, 128]
		out_dim = k[-1]//(len(k)*len(k))

		layers = [
		# Input shape = [B, T, F]
		Residual_Net([self.T, self.F], self.training, [1, 32, 64, 128], 3),
		# Output shape = [B, T/4, F/4, 128]
		Reshape([-1, k[-1]/(len(k)*len(k))]),
		#Conv1D([1, k[-1], 4*self.E]),
		Dense(k[-1]/(len(k)*len(k)), self.E),
		# Output shape = [B, T/4, F/4, 4*E]
		Reshape([-1, self.T, self.F, self.E])
		# Output shape = [B, T, F, E]
		]

		def f_props(layers, x):
			for i, layer in enumerate(layers):
				print layer.name
				x = layer.f_prop(x)
				print x.shape
			return x

		y = f_props(layers, tf.expand_dims(self.X,3))

		return y

	@ops.scope
	def cost(self):
		# Definition of cost for DAS model

		# V [B, T, F, E]
		V = self.prediction
		V = tf.expand_dims(V, 3)
		# Now V [B, T, F, 1, E]

		# U [M, E]
		Ind = tf.expand_dims(self.Ind,2)

		U = tf.gather_nd(self.speaker_centroids, Ind)
		U = tf.expand_dims(U,1)
		U = tf.expand_dims(U,1)
		# Now U [1, 1, 1, M, E]

		# W [B, T, F]
		Ws = tf.expand_dims(self.Ws,3)
		Ws = tf.expand_dims(Ws,3)
		# Now W [B, T, F, 1, 1]

		prod = tf.reduce_sum(Ws * V * U, 4)


		cost = - tf.log(tf.nn.sigmoid(self.Y * prod))

		cost = tf.reduce_mean(cost, 3)
		cost = tf.reduce_mean(cost, 0)
		cost = tf.reduce_mean(cost)

		centroids_cost = tf.nn.l2_loss(tf.matmul(self.speaker_centroids,tf.transpose(self.speaker_centroids)))
		cost = cost + self.l * centroids_cost

		tf.summary.scalar('cost', cost)

		return cost

	@ops.scope
	def optimize(self):
		return tf.train.AdamOptimizer().minimize(self.cost)

	def train(self, X_train, Y_train, Ind_train, step):
		cost, _, summary = self.sess.run([self.cost, self.optimize, self.merged],
			{self.X: X_train, self.Y: Y_train, self.Ind:Ind_train, self.training : True})
		self.train_writer.add_summary(summary, step)
		return cost


