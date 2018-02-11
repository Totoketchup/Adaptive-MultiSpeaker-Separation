# -*- coding: utf-8 -*-
from utils.ops import scope, AMSGrad
import os
import config
import tensorflow as tf
import haikunator
from itertools import compress
import numpy as np
import json

class Network(object):
	"""docstring for Network"""
	def __init__(self, *args, **kwargs):

		# Constant seed for uniform results
		tf.set_random_seed(42)
		np.random.seed(42)

		##
		## Model Configuration 
		if kwargs is not None:
			self.folder = kwargs['type']
			self.optimizer = kwargs['optimizer']
			self.S = kwargs['nb_speakers']
			self.args = kwargs
		else:
			raise Exception('Keyword Arguments missing ! Please add the right arguments in input | check doc')

		# Run ID
		self.runID = haikunator.Haikunator().haikunate()
		print 'ID : {}'.format(self.runID)

		#Create a graph for this model
		self.graph = tf.Graph()

		with self.graph.as_default():

			with tf.name_scope('inputs'):

				# Boolean placeholder signaling if the model is in learning/training mode
				self.training = tf.placeholder(tf.bool, name='is_training')

				# Placeholder for the learning rate
				self.learning_rate = tf.placeholder("float", name='learning_rate')

				# Batch of raw non-mixed audio
				# shape = [ batch size , number of speakers, samples ] = [ B, S, L]
				self.x_non_mix = tf.placeholder("float", [None, None, None], name='non_mix_input')

				# Batch of raw mixed audio - Input data
				# shape = [ batch size , samples ] = [ B , L ]
				self.x_mix = tf.placeholder("float", [None, None], name='mix_input')

				# Speakers indicies used in the mixtures
				# shape = [ batch size, #speakers]
				self.Ind = tf.placeholder(tf.int32, [None, None], name='indicies')

				shape_in = tf.shape(self.x_mix)
				self.B = shape_in[0]
				self.L = shape_in[1]

		# Create a session for this model based on the constructed graph
		config_ = tf.ConfigProto()
		config_.gpu_options.allow_growth = True
		config_.allow_soft_placement = True
		self.sess = tf.Session(graph=self.graph, config=config_)

	def tensorboard_init(self):
		with self.graph.as_default():
			self.saver = tf.train.Saver()
			self.merged = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(os.path.join(config.log_dir,self.folder,self.runID,'train'), self.graph)
			self.valid_writer = tf.summary.FileWriter(os.path.join(config.log_dir,self.folder,self.runID,'valid'))

			# Save arguments
			with open(os.path.join(config.log_dir,self.folder,self.runID,'params'), 'w') as f:
				json.dump(self.args, f)

	def create_saver(self, subset=None):
		with self.graph.as_default():
			if subset is None:
				self.saver = tf.train.Saver()
			else:
				self.saver = tf.train.Saver(subset)

	# Restore last checkpoint of the current graph using the total path
	# This method is used when we plug a new layer
	def restore_model(self, path):
		self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
	
	# Restore the last checkpoint of the current trained model
	# This function is maintly used during the test phase
	def restore_last_checkpoint(self):
		self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(config.log_dir, self.folder ,self.runID)))

	def init_all(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())
 
	def non_initialized_variables(self):
		with self.graph.as_default():
			global_vars = tf.global_variables()
			is_not_initialized = self.sess.run([~(tf.is_variable_initialized(var)) \
										   for var in global_vars])
			not_initialized_vars = list(compress(global_vars, is_not_initialized))
			print 'not init: '
			print [v.name for v in not_initialized_vars]
			if len(not_initialized_vars):
				init = tf.variables_initializer(not_initialized_vars)
				return init

	def initialize_non_init(self):
		with self.graph.as_default():
			self.sess.run(self.non_initialized_variables())

	@scope
	def optimize(self):
		print 'Train the following variables :'
		print self.trainable_variables

		optimizer = AMSGrad(self.learning_rate, epsilon=0.001)
		gradients, variables = zip(*optimizer.compute_gradients(self.cost, var_list=self.trainable_variables))
		optimize = optimizer.apply_gradients(zip(gradients, variables))
		return optimize

	def save(self, step):
		path = os.path.join(config.log_dir, self.folder ,self.runID, "model")
		self.saver.save(self.sess, path, step)
		return path

	def train(self, X_mix, X_non_mix, learning_rate, step, I):
		summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.x_mix: X_mix, self.x_non_mix:X_non_mix, self.training:True, self.Ind:I, self.learning_rate:learning_rate})
		self.train_writer.add_summary(summary, step)
		return cost

	def valid_batch(self, X_mix_valid, X_non_mix_valid, I):
		return self.sess.run(self.cost, {self.x_non_mix:X_non_mix_valid, self.x_mix:X_mix_valid, self.training:False, self.Ind:I})

	def add_valid_summary(self, val, step):
		summary = tf.Summary()
		summary.value.add(tag="Valid Cost", simple_value=val)
		self.valid_writer.add_summary(summary, step)

	def freeze_all_with(self, prefix):
		to_delete = []
		for var in self.trainable_variables:
			if prefix in var.name:
				to_delete.append(var)
		for to_del in to_delete:
			self.trainable_variables.remove(to_del)

	def freeze_all_except(self, prefix):
		to_train = []
		for var in self.trainable_variables:
			if prefix in var.name:
				to_train.append(var)
		self.trainable_variables = to_train

	@staticmethod
	def load(path, modified_args):
		# Load parameters used for the desired model to load
		params_path = os.path.join(path, 'params')
		with open(params_path) as f:
			args = json.load(f)
		# Update with new args such as 'pretraining' or 'type'
		args.update(modified_args)

		# Create a new Adapt model with these parameters
		return Network(**args)

		