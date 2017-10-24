# My Model 
from utils.ops import ops
import tensorflow as tf

#############################################
#       Dense Separator Model       #
#############################################

class Dense_net:

	def __init__(self, input_tensor, adapt_front):
		self.adapt_model = adapt_front
		self.B = adapt_front.B
		self.S = adapt_front.S
		self.dim = adapt_front.N*40
		with adapt_front.graph.as_default():
			# # Batch of spectrogram chunks - Input data
			# # shape = [ batch size , chunk size, F ]
			# and
			# # Batch of raw spectrogram chunks - Input data
			# # shape = [ batch size , samples ]
			self.X, self.X_raw = input_tensor

	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	@ops.scope
	def prediction(self):
		# DAS network

		shape = tf.shape(self.X)
		T = shape[1]
		F = shape[2]

		out = tf.layers.dense(tf.reshape(self.X,[self.B,self.dim]), 500, tf.nn.relu)
		out = tf.layers.dense(out, 500, tf.nn.relu)
		out = tf.layers.dense(out, self.dim)
		out = tf.reshape(out, shape)

		sep_2 = tf.abs(self.X - out)
		output = tf.reshape(tf.concat([out, sep_2], axis=0), [self.B*self.S, T, F, 1])

		print output
		return output

