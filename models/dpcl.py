# -*- coding: utf-8 -*-
from utils.ops import BLSTM, Conv1D, Reshape, Normalize, f_props, scope, AMSGrad
from utils.tools import args_to_string
import haikunator
from models.Kmeans_2 import KMeans
import os
import config
import tensorflow as tf

############################################
#       Deep Clustering Architecture       #
############################################

class DPCL:

	def __init__(self, runID=None, **kwargs):

		if runID is not None:
			self.F = kwargs['window_size']
			self.layer_size = kwargs['layer_size']
			self.embedding_size = kwargs['embedding_size']
			self.nonlinearity = kwargs['nonlinearity']
			self.normalize = kwargs['normalize']
			self.B = kwargs['B']
			self.S = kwargs['nb_speakers']
			self.adapt_front = kwargs['front']

			self.graph = self.adapt_front.graph

			with self.graph.as_default():
				self.X, self.X_non_mix = kwargs['input_tensor']
				with tf.name_scope('create_masks'):
					self.Y = tf.one_hot(tf.argmax(self.X_non_mix, axis=3), 2, 1.0, 0.0)

		else:

			#Create a graph for this model
			self.graph = tf.Graph()

			with self.graph.as_default():
				# Global params
				self.folder = kwargs['type']

				# STFT hyperparams
				self.window_size = kwargs['window_size']
				self.hop_size = kwargs['hop_size']

				# Network hyperparams
				self.F = kwargs['freqs_size']
				self.layer_size = kwargs['layer_size']
				self.embedding_size = kwargs['embedding_size']
				self.normalize = kwargs['normalize']
				self.learning_rate = kwargs['learning_rate']

				# Run ID for tensorboard
				self.runID = 'DPCL_STFT' + '-' + haikunator.Haikunator().haikunate()
				print 'ID : {}'.format(self.runID)
				if kwargs is not None:
					self.runID += args_to_string(kwargs)

				# Placeholder tensor for the mixed signals
				self.x_mix = tf.placeholder("float", [None, None])

				# Placeholder tensor for non mixed input data [B, T, F, S]
				# Place holder for non mixed signals [B, S, L]
				self.x_non_mix = tf.placeholder("float", [None, None, None])

				self.preprocessing
				self.normalization01
				self.prediction
				self.cost
				self.optimize

				config_ = tf.ConfigProto()
				config_.gpu_options.allow_growth = True
				config_.allow_soft_placement = True
				self.sess = tf.Session(graph=self.graph, config=config_)

	def tensorboard_init(self):
		with self.graph.as_default():
			self.merged = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, self.folder, self.runID, 'train'), self.graph)
			self.valid_writer = tf.summary.FileWriter(os.path.join(config.log_dir, self.folder, self.runID, 'valid'), self.graph)
			self.saver = tf.train.Saver()

	def save(self, step):
		path = os.path.join(config.log_dir, self.folder ,self.runID ,"model.ckpt")
		self.saver.save(self.sess, path, step)
		return path

	def restore_last_checkpoint(self):
		self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(config.log_dir, self.folder ,self.runID)))


	def init(self):
			with self.graph.as_default():
				self.sess.run(tf.global_variables_initializer())


	@scope
	def normalization01(self):
		min_ = tf.reduce_min(self.X, axis=[1,2], keep_dims=True)
		max_ = tf.reduce_max(self.X, axis=[1,2], keep_dims=True)
		self.X = (self.X - min_) / (max_ - min_)

	@scope
	def normalization_mean_std(self):
		mean, var = tf.nn.moments(self.X, axes=[1,2], keep_dims=True)
		self.X = (self.X - mean) / var


	@scope
	def preprocessing(self):
		self.stfts = tf.contrib.signal.stft(self.x_mix, 
			frame_length=self.window_size, 
			frame_step=self.window_size-self.hop_size,
			fft_length=self.window_size)

		self.B = tf.shape(self.x_non_mix)[0]
		self.S = tf.shape(self.x_non_mix)[1]

		self.stfts_non_mix = tf.contrib.signal.stft(tf.reshape(self.x_non_mix, [self.B*self.S, -1]), 
			frame_length=self.window_size, 
			frame_step=self.window_size-self.hop_size,
			fft_length=self.window_size)

		self.X = tf.sqrt(tf.abs(self.stfts))
		self.X_non_mix = tf.sqrt(tf.abs(self.stfts_non_mix))
		self.X_non_mix = tf.reshape(self.X_non_mix, [self.B, self.S, -1, self.F])
		self.X_non_mix = tf.transpose(self.X_non_mix, [0, 2, 3, 1])

		argmax = tf.argmax(tf.abs(self.X_non_mix), axis=3)
		self.Y = tf.one_hot(argmax, 2, 1.0, 0.0)


	@scope
	def prediction(self):
		# DPCL network

		shape = tf.shape(self.X)

		layers = [
			BLSTM(self.layer_size, 'BLSTM_1'),
			BLSTM(self.layer_size, 'BLSTM_2'),
			Conv1D([1, self.layer_size, self.embedding_size*self.F]),
			Reshape([self.B, shape[1], self.F, self.embedding_size]),
			Normalize(3)
		]

		y = f_props(layers, self.X)
		
		return y

	@scope
	def cost(self):
		# Definition of cost for DAS model

		# Get the shape of the input
		shape = tf.shape(self.Y)
		B = shape[0]
		T = shape[1]
		F = shape[2]
		S = shape[3]

		# Reshape the targets to be of shape (batch, T*F, c) and the vectors to
		# have shape (batch, T*F, K)
		Y = tf.reshape(self.Y, [B, T*F, S])
		V = tf.reshape(self.prediction, [B, T*F, self.embedding_size])

		# Compute the partition size vectors
		ones = tf.ones([B, T*F, 1])
		mul_ones = tf.matmul(tf.transpose(Y, perm=[0,2,1]), ones)
		diagonal = tf.matmul(Y, mul_ones)
		D = 1/tf.sqrt(diagonal)
		D = tf.reshape(D, [B, T*F])

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
		tf.summary.scalar('1', tf.reduce_mean(tf.norm(VTV, axis=[-2,-1])))
		tf.summary.scalar('2', tf.reduce_mean(-2*tf.norm(VTY, axis=[-2,-1])))
		tf.summary.scalar('3', tf.reduce_mean(tf.norm(YTY, axis=[-2,-1])))

		return cost

	@scope
	def separate(self):
		# Input for KMeans algorithm [B, TF, E]
		input_kmeans = tf.reshape(self.prediction, [self.B, -1, self.embedding_size])
		# S speakers to separate, give self.X in input not to consider silent bins
		kmeans = KMeans(nb_clusters=self.S, nb_iterations=10, input_tensor=input_kmeans, latent_space_tensor=self.X)
		
		# Extract labels of each bins TF_i - labels [B, TF, 1]
		_ , labels = kmeans.network
		self.masks = tf.one_hot(labels, 2, 1.0, 0.0) # Create masks [B, TF, S]

		separated = tf.reshape(self.X, [self.B, -1, 1])* self.masks # [B ,TF, S] 
		separated = tf.reshape(separated, [self.B, -1, self.F, self.S])
		separated = tf.transpose(separated, [0,3,1,2]) # [B, S, T, F]
		separated = tf.reshape(separated, [self.B*self.S, -1, self.F, 1]) # [BS, T, F, 1]

		return separated

	@scope
	def optimize(self):
		# optimizer = self.select_optimizer(self.optimizer)(self.learning_rate)
		optimizer = AMSGrad(self.learning_rate, epsilon=0.001)
		opt = optimizer.minimize(self.cost)
		return opt

	def train(self, X_mix, X_non_mix, step):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.x_mix: X_mix, self.x_non_mix:X_non_mix})
		# cost = self.sess.run(self.stfts, {self.x_mix: X_mix, self.x_non_mix:X_non_mix, self.I:I})
		self.train_writer.add_summary(summary, step)
		return cost

	def valid_batch(self, X_mix_valid, X_non_mix_valid):
		cost = self.sess.run(self.cost, {self.x_non_mix:X_non_mix_valid, self.x_mix:X_mix_valid})
		return cost

	def add_valid_summary(self, val, step):
		summary = tf.Summary()
		summary.value.add(tag="Valid Cost", simple_value=val)
		self.valid_writer.add_summary(summary, step)
