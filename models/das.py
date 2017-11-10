# My Model 
from utils.ops import ops
from utils.ops.ops import Residual_Net, Reshape, Dense, f_props
from tensorflow.contrib.tensorboard.plugins import projector
# from utils.postprocessing.reconstruction import 
from models.Kmeans_2 import KMeans
import os
import config
import tensorflow as tf

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class DAS:

	def __init__(self, input_tensor, adapt_front, S_tot=config.dev_clean_speakers, E=config.embedding_size, threshold=config.threshold, l=0.2):

		self.B = adapt_front.B
		self.F = adapt_front.N    # Freqs size
		self.E = E          # Embedding size
		self.S_tot = S_tot          # Total number of speakers
		self.threshold = threshold # Threshold for silent weights
		self.l = l

		with tf.get_default_graph().as_default():

			# # Batch of spectrogram chunks - Input data
			# # shape = [ batch size , chunk size, F ]
			# and
			# # Batch of raw spectrogram chunks - Input data
			# # shape = [ batch size , samples ]
			self.X, self.X_raw = input_tensor
			with tf.name_scope('create_masks'):
				# # Batch of Masks (bins label)
				# # shape = [ batch size, chunk size, F, #speakers ]
				argmax = tf.argmax(self.X_raw, axis=3)
				self.Y = tf.one_hot(argmax, 2, 1.0, -1.0)

			# Speakers indicies used in the mixtures
			# shape = [ batch size, #speakers]
			self.Ind = adapt_front.Ind

			# Placeholder for the 'dropout', telling if the network is 
			# currently learning or not
			self.training = adapt_front.training

			self.Ws = tf.cast(self.X > 0, self.X.dtype) * self.X

			# The centroids used for each speaker
			# shape = [ #tot_speakers, embedding size]
			self.speaker_centroids = tf.Variable(
				tf.truncated_normal([self.S_tot,self.E], 
				stddev=tf.sqrt(2/float(self.E))),
				name='centroids')

	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	@ops.scope
	def prediction(self):
		# DAS network

		shape = tf.shape(self.X)
		T = shape[1]
		F = shape[2]

		k = [1, 32, 64, 128]
		# out_dim = k[-1]//(len(k)*len(k))
		# F_out = int(self.F/pow(2,2)) + self.F%(self.F/pow(2,2))

		layers = [
		Residual_Net([T, F], self.training, k, len(k)-1),
		Reshape([-1, k[-1]]),
		Dense(k[-1], self.E),
		Reshape([shape[0], T, self.F, self.E])
		]

		y = f_props(layers, tf.expand_dims(self.X,3))

		return y

	@ops.scope
	def cost(self):
		# Definition of cost for DAS model

		# V [B, T, F, E]
		V = self.prediction
		V = tf.expand_dims(V, 3)
		# Now V [B, T, F, 1, E]

		# Ind [B, S, 1]
		Ind = tf.expand_dims(self.Ind, 2)

		# U [S_tot, E]
		U = tf.gather_nd(self.speaker_centroids, Ind)
		U = tf.expand_dims(U,1)
		U = tf.expand_dims(U,1)
		# Now U [1, 1, 1, M, E]

		# W [B, T, F]
		Ws = tf.expand_dims(self.Ws,3)
		Ws = tf.expand_dims(Ws,3)
		# Now W [B, T, F, 1, 1]

		prod = tf.reduce_sum(V * U * Ws, 4)

		test = tf.identity(prod, name='test')
		# centroids_cost = tf.nn.l2_loss(tf.matmul(self.speaker_centroids,tf.transpose(self.speaker_centroids)))

		cost = - tf.log(tf.nn.sigmoid(self.Y * prod)) #-  self.l *centroids_cost
		self.tt = V
		cost = tf.reduce_mean(cost, 3)
		cost = tf.reduce_mean(cost, 0)
		cost = tf.reduce_mean(cost)

		tf.summary.scalar('training_cost', cost)

		return cost

	@ops.scope
	def separate(self):
		# TODO for only when Ind is available, speaker info is given
		# Ind [B, S, 1]
		Ind = tf.expand_dims(self.Ind, 2)

		# U [S_tot, E]
		centroids = tf.gather_nd(self.speaker_centroids, Ind)

		input_kmeans = tf.reshape(self.prediction, [self.B, -1, self.E])
		kmeans = KMeans(nb_clusters=2, nb_iterations=10, input_tensor=input_kmeans, centroids_init=centroids)
		_ , labels = kmeans.network
		
		masks = tf.one_hot(labels, 2, 1.0, 0.0)
		separated = tf.reshape(self.X, [self.B, -1, 1])* masks # [B ,TF, S] 
		separated = tf.reshape(separated, [self.B, -1, self.F, self.S])
		separated = tf.transpose(separated, [0,3,1,2])
		separated = tf.reshape(separated, [self.B*self.S, -1, self.F, 1])
		print separated
		return separated

	@ops.scope
	def training_cost(self):
		cost_train = self.cost
		tf.summary.scalar('training cost', cost_train)
		return cost_train

	@ops.scope
	def validation_cost(self):
		cost_valid = self.valid_cost
		tf.summary.scalar('validation cost', cost_valid)
		return cost_valid


	@ops.scope
	def optimize(self):
		return tf.train.AdamOptimizer(0.001).minimize(self.cost)

	def train(self, X_train, Y_train, Ind_train, x, step):
		cost, other,_, summary = self.sess.run([self.training_cost, self.tt, self.optimize, self.merged],
			{self.X: X_train, self.Y: Y_train, self.Ind:Ind_train, self.X_raw: x, self.training : True})
		self.train_writer.add_summary(summary, step)
		return cost, other


	def save(self, step):
		self.saver.save(self.sess, os.path.join('log/', "deep_adaptive_separator_model.ckpt"))#, step)

	def embeddings(self, X):
		V = self.sess.run(self.prediction, {self.X: X, self.training: False})
		return V

	def valid(self, X, X_raw, Y, I, step):
		cost, summary = self.sess.run([self.validation_cost, self.merged], {self.X: X, self.Y: Y, self.Ind:I, self.X_raw:X_raw, self.training: False} )
		self.train_writer.add_summary(summary, step)
		return cost



