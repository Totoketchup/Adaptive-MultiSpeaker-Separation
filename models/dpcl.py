# My Model 
from utils.ops import ops
from utils.ops.ops import BLSTM, Conv1D, Reshape, Normalize
from tensorflow.contrib.tensorboard.plugins import projector
# from utils.postprocessing.reconstruction import 

import os
import config
import tensorflow as tf

#############################################
#       Deep Adaptive Separator Model       #
#############################################

class DPCL:

	def __init__(self, fftsize=config.fftsize//2, E=config.embedding_size, threshold=config.threshold):

		self.F = fftsize    # Freqs size
		self.E = E          # Embedding size
		self.threshold = threshold # Threshold for silent weights
		
		self.graph = tf.Graph()

		with self.graph.as_default():
			# Batch of spectrogram chunks - Input data
			# shape = [ batch size , chunk size, F ]
			self.X = tf.placeholder("float", [None, None, self.F])

			# Batch of spectrogram chunks - Input data
			# shape = [ batch size , samples ]
			self.X_raw = tf.placeholder("float", [None, None])

			# Batch of Masks (bins label)
			# shape = [ batch size, chunk size, F, #speakers ]
			self.Y = tf.placeholder("float", [None, None, self.F, None])

			self.Ws = tf.cast(self.X - threshold > 0, self.X.dtype) * self.X

			self.audio_writer = tf.summary.audio(name= "input", tensor = self.X_raw, sample_rate = config.fs)
			# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
			config_ = projector.ProjectorConfig()

			# You can add multiple embeddings. Here we add only one.
			self.embedding = config_.embeddings.add()

			self.prediction
			self.cost
			self.optimize

			self.saver = tf.train.Saver()
			self.merged = tf.summary.merge_all()

			# Link this tensor to its metadata file (e.g. labels).

			self.train_writer = tf.summary.FileWriter('log/', self.graph)

			# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
			# read this file during startup.
			projector.visualize_embeddings(self.train_writer, config_)


		# Create a session for this model based on the constructed graph
		self.sess = tf.Session(graph = self.graph)


	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	@ops.scope
	def prediction(self):
		# DAS network

		shape = tf.shape(self.X)

		layers = [
		# Input shape = [B, T, F]
		BLSTM(600, 'BLSTM_1'),
		BLSTM(600, 'BLSTM_2'),
		# Input shape = [B, T, F, 600]
		Conv1D([1, 600, self.E*self.F]),
		Reshape([shape[0], shape[1], self.F, self.E]),
		Normalize(3)
		]

		def f_props(layers, x):
			for i, layer in enumerate(layers):
				print layer.name
				x = layer.f_prop(x)
				print x.shape
			return x

		y = f_props(layers, self.X)
		
		self.embedding.tensor_name = y.name

		return y

	@ops.scope
	def cost(self):
		# Definition of cost for DAS model

		# Get the shape of the input
		shape = tf.shape(self.Y)

		# Reshape the targets to be of shape (batch, T*F, c) and the vectors to
		# have shape (batch, T*F, K)
		Y = tf.reshape(self.Y, [shape[0], shape[1]*shape[2], shape[3]])
		V = tf.reshape(self.prediction, [shape[0], shape[1]*shape[2], self.E])

		# Compute the partition size vectors
		ones = tf.ones([shape[0], shape[1]*shape[2], 1])
		mul_ones = tf.matmul(tf.transpose(Y, perm=[0,2,1]), ones)
		diagonal = tf.matmul(Y, mul_ones)
		D = 1/tf.sqrt(diagonal)
		D = tf.reshape(D, [shape[0], shape[1]*shape[2]])

		# Compute the matrix products needed for the cost function.  Reshapes
		# are to allow the diagonal to be multiplied across the correct
		# dimensions without explicitly constructing the full diagonal matrix.
		DV  = D * tf.transpose(V, perm=[2,0,1])
		DV = tf.transpose(DV, perm=[1,2,0])
		VTV = tf.matmul(tf.transpose(V, perm=[0,2,1]), DV)

		DY = D * tf.transpose(Y, perm=[2,0,1])
		DY = tf.transpose(DY, perm=[1,2,0])
		VTY = tf.matmul(tf.transpose(V, perm=[0,2,1]), DY)

		YTY = tf.matmul(tf.transpose(Y, perm=[0,2,1]), DY)

		# Compute the cost by taking the Frobenius norm for each matrix
		cost = tf.norm(VTV, axis=[-2,-1]) -2*tf.norm(VTY, axis=[-2,-1]) + tf.norm(YTY, axis=[-2,-1])

		cost = tf.reduce_mean(cost)

		tf.summary.scalar('cost', cost)
		
		return cost


	@ops.scope
	def optimize(self):
		return tf.train.AdamOptimizer().minimize(self.cost)

	def train(self, X_train, Y_train, x, step):
		cost, _, summary = self.sess.run([self.cost, self.optimize, self.merged],
			{self.X: X_train, self.Y: Y_train, self.X_raw: x})
		self.train_writer.add_summary(summary, step)
		return cost


	def save(self, step):
		self.saver.save(self.sess, os.path.join('log/', "deep_clustering_model.ckpt"), step)

	def embeddings(self, X):
		V = self.sess.run(self.prediction, {self.X: X})
		return V

