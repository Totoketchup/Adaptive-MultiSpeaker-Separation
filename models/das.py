# My Model 
from utils.ops import ops
from utils.ops.ops import Residual_Net, Conv1D, Reshape, Dense, f_props
from tensorflow.contrib.tensorboard.plugins import projector
# from utils.postprocessing.reconstruction import 

import os
import config
import tensorflow as tf
import time 

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class DAS:

	def __init__(self,  graph=None, S=config.dev_clean_speakers, fftsize=config.fftsize//2 + 1, E=config.embedding_size, threshold=config.threshold, l=0.2):

		self.F = fftsize    # Freqs size
		self.E = E          # Embedding size
		self.S = S          # Total number of speakers
		self.threshold = threshold # Threshold for silent weights
		self.l = l
		
		if graph == None:
			self.graph = tf.Graph()
		else:
			self.graph = graph


		with self.graph.as_default():
			# Batch of spectrogram chunks - Input data
			# shape = [ batch size , chunk size, F ]
			self.X = tf.placeholder("float", [None, None, None])

			# Batch of spectrogram chunks - Input data
			# shape = [ batch size , samples ]
			self.X_raw = tf.placeholder("float", [None, None])

			# Batch of Masks (bins label)
			# shape = [ batch size, chunk size, F, #speakers ]
			self.Y = tf.placeholder("float", [None, None, self.F, None])

			# Speakers indicies used in the mixtures
			# shape = [ batch size, #speakers]
			self.Ind = tf.placeholder(tf.int32, [None,None])

			# Placeholder for the 'dropout', telling if the network is 
			# currently learning or not
			self.training = tf.placeholder(tf.bool)

			self.Ws = tf.cast(self.X > 0, self.X.dtype) * self.X

			# The centroids used for each speaker
			# shape = [ #tot_speakers, embedding size]
			self.speaker_centroids = tf.Variable(
				tf.truncated_normal([self.S,self.E], 
				stddev=tf.sqrt(2/float(self.E))),
				name='centroids')

			self.audio_writer = tf.summary.audio(name= "input", tensor = self.X_raw, sample_rate = config.fs)

			self.prediction
			self.training_cost
			self.optimize

			self.saver = tf.train.Saver()
			self.merged = tf.summary.merge_all()

			# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
			config_ = projector.ProjectorConfig()

			# You can add multiple embeddings. Here we add only one.
			embedding = config_.embeddings.add()
			embedding.tensor_name = self.speaker_centroids.name
			# Link this tensor to its metadata file (e.g. labels).

			self.train_writer = tf.summary.FileWriter('log/', self.graph)

			# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
			# read this file during startup.
			projector.visualize_embeddings(self.train_writer, config_)


		# Create a session for this model based on the constructed graph
		self.sess = tf.Session(graph = self.graph)

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
			self.training = True # TODO

			# self.Ws = tf.cast(self.X > 0, self.X.dtype) * self.X

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

		# U [M, E]
		Ind = tf.expand_dims(self.Ind, 2)

		U = tf.gather_nd(self.speaker_centroids, Ind)
		U = tf.expand_dims(U,1)
		U = tf.expand_dims(U,1)
		# Now U [1, 1, 1, M, E]

		# W [B, T, F]
		# Ws = tf.expand_dims(self.Ws,3)
		# Ws = tf.expand_dims(Ws,3)
		# Now W [B, T, F, 1, 1]

		prod = tf.reduce_sum(V * U, 4)
		centroids_cost = tf.nn.l2_loss(tf.matmul(self.speaker_centroids,tf.transpose(self.speaker_centroids)))
		
		cost = - tf.log(tf.nn.sigmoid(self.Y * prod)) #-  self.l *centroids_cost
		self.tt = V
		cost = tf.reduce_mean(cost, 3)
		cost = tf.reduce_mean(cost, 0)
		cost = tf.reduce_mean(cost)

		tf.summary.scalar('training_cost', cost)

		return cost

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



