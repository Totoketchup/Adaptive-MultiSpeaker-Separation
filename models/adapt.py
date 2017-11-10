# -*- coding: utf-8 -*-
# My Model 
from utils.ops import ops
from utils.ops.ops import unpool, variable_summaries, get_scope_variable
from itertools import permutations
# from tensorflow.contrib.tensorboard.plugins import projector
# from utils.postprocessing.reconstruction import 
# import matplotlib.pyplot as plt
import os
import config
import tensorflow as tf
import haikunator
from itertools import compress
from tensorflow.python.saved_model import builder as saved_model_builder
name = 'AdaptiveNet'

#############################################
#     Adaptive Front and Back End Model     #
#############################################

class Adapt:
	def __init__(self, config_model=None, N=256, maxpool=256, l=0.001, pretraining=True, runID=None, separator=None, folder='default'):
		
		self.max_pool_value = maxpool
		self.l = l
		self.pretraining = pretraining
		self.sepNet = separator
		self.folder = folder
		self.smooth_size = 4.0
		self.p = 0.05
		self.beta = 1.0
		self.same_filter = False

		##
		## Model Configuration 
		##
		if config_model != None:
			self.N = config_model['N']
			self.max_pool_value = config_model['maxpool']
			self.l = config_model['reg']
			self.folder = config_model['type']
			self.smooth_size = config_model['smooth_size']
			self.beta = config_model['beta']
			self.p = config_model['rho']
			self.window = config_model['window']
			self.same_filter = config_model['same_filter']
			self.optimizer = config_model['optimizer']

		if runID == None:
			# Run ID for tensorboard
			self.runID = name + '-' + haikunator.Haikunator().haikunate()
			print 'ID : {}'.format(self.runID)
			if config_model != None:
				self.runID += ''.join('-{}={}-'.format(key, val) for key, val in sorted(config_model.items()))
		else:
			self.runID = name + '-' + runID



		#Create a graph for this model
		self.graph = tf.Graph()

		with self.graph.as_default():

			# Boolean placeholder signaling if the model is in learning/training mode
			self.training = tf.placeholder(tf.bool, name='is_training')

			# Placeholder for the learning rate
			self.learning_rate = tf.placeholder("float", name='learning_rate')

			# Batch of raw non-mixed audio
			# shape = [ batch size , number of speakers, samples ] = [ B, S, L]
			self.X_non_mix = tf.placeholder("float", [None, None, None], name='non_mix_input')
			
			self.S = 2 # 2 speakers used # TODO

			if pretraining:
				shape_in = tf.shape(self.X_non_mix)
				self.B = shape_in[0]
				self.L = shape_in[2]

				self.x = tf.reshape(self.X_non_mix, [self.B*self.S, self.L])
				self.B_tot = self.B*self.S
			else:

				# Batch of raw mixed audio - Input data
				# shape = [ batch size , samples ] = [ B , L ]
				self.X_mix = tf.placeholder("float", [None, None], name='mix_input')

				# Speakers indicies used in the mixtures
				# shape = [ batch size, #speakers]
				self.Ind = tf.placeholder(tf.int32, [None,None], name='indicies')
				shape_in = tf.shape(self.X_mix)

				self.B = shape_in[0]
				self.L = shape_in[1]

				self.x = tf.cond(self.training, 
					lambda: tf.concat([self.X_mix,  tf.reshape(self.X_non_mix, [self.B*self.S, self.L])], axis=0),
					lambda: self.X_mix
					)
				self.B_tot = tf.cond(self.training, 
					lambda: self.B*(self.S+1),
					lambda: self.B
					)

			if pretraining:
				self.front
				self.separator
				self.back
				self.cost
				self.optimize
			else:
				self.front
			
		# Create a session for this model based on the constructed graph
		config_ = tf.ConfigProto()
		config_.gpu_options.allow_growth = True
		config_.allow_soft_placement = True
		# config_.log_device_placement = True
		self.sess = tf.Session(graph=self.graph, config=config_)


	def tensorboard_init(self):
		with self.graph.as_default():

			# with tf.name_scope('conv'):
			# 	variable_summaries(self.W)
			# 	if not self.same_filter : variable_summaries(self.WT)

			# variable_summaries(self.window_filter)
			# variable_summaries(self.bases)
			# variable_summaries(self.conv_filter)

			# variable_summaries(self.smoothing_filter)

			# tf.summary.scalar('learning_rate', self.learning_rate)

			tf.summary.audio(name= "input/", tensor = self.x, sample_rate = config.fs, max_outputs=2)

			tf.summary.audio(name= "output/reconstructed", tensor = tf.reshape(self.back, [-1, self.L]), sample_rate = config.fs)

			# trs = lambda x : tf.transpose(x, [0, 2, 1, 3])
			# tf.summary.image(name= "stft_like", tensor = trs(self.y))

			# tf.summary.image(name= "separated_stft", tensor = trs(self.separator[0]))

			# tf.summary.image(name= "unpooled", tensor = trs(self.unpooled))
			# tf.summary.image(name= "mask", tensor = trs(self.mask))
			# tf.summary.image(name= "reconstructed", tensor = trs(self.recons))
			# if self.pretraining:
			with tf.name_scope('loss_values'):
				tf.summary.scalar('loss', self.MSE)
			# 	tf.summary.scalar('sparsity', tf.reduce_mean(self.p_hat))
			# 	tf.summary.scalar('sparse_reg', self.sparse_reg)
			# 	tf.summary.scalar('regularization', self.reg)
			# 	tf.summary.scalar('training_cost', self.cost)

			self.merged = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(os.path.join(config.log_dir,self.folder,self.runID), self.graph)
			self.saver = tf.train.Saver()

		# if self.sepNet != None:
		# 	tf.summary.image(name="back", tensor= trs(self.separator[0]))
		# 	tf.summary.image(name="mask_1", tensor= trs(self.sepNet.ms[:,:,:,0:1]))
		# 	tf.summary.image(name="mask_2", tensor= trs(self.sepNet.ms[:,:,:,1:]))

			# config_ = projector.ProjectorConfig()
			# embedding = config_.embeddings.add()
			# embedding.tensor_name = self.sepNet.embedding.name
			# projector.visualize_embeddings(self.train_writer, config_)


	def create_saver(self):
		with self.graph.as_default():
			self.saver = tf.train.Saver()

	def restore_model(self, path, runID):
		self.saver.restore(self.sess, os.path.join(path, name+'-'+runID, 'model.ckpt'))

	def savedModel(self):
		with self.graph.as_default():
			path = os.path.join(config.log_dir,self.folder ,self.runID, 'SavedModel')
			builder = saved_model_builder.SavedModelBuilder(path)

			# Build signatures
			input_tensor_info = tf.saved_model.utils.build_tensor_info(self.X_mix)
			output_tensor_info = tf.saved_model.utils.build_tensor_info(self.back)

			signature = tf.saved_model.signature_def_utils.build_signature_def(
				inputs={
					'mixed_audio':
					input_tensor_info
				},
				outputs={
					'unmixed_audio':
					output_tensor_info
				}
				)


			builder.add_meta_graph_and_variables(
				self.sess, ['validating'],
				signature_def_map={'separate_audio':signature}
				)

			builder.save()
			print 'Successfully exported model to %s' % path



	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())
 
	def non_initialized_variables(self):
		with self.graph.as_default():
			global_vars = tf.global_variables()
			is_not_initialized = self.sess.run([~(tf.is_variable_initialized(var)) \
										   for var in global_vars])
			not_initialized_vars = list(compress(global_vars, is_not_initialized))
			if len(not_initialized_vars):
				init = tf.variables_initializer(not_initialized_vars)
				return init

	def connect_only_front_to_separator(self, separator, freeze_front=True):
		with self.graph.as_default():
			self.connect_front(separator)
			self.sepNet.output = self.sepNet.prediction
			self.cost = self.sepNet.cost
			if freeze_front:
				self.freeze_front()
			self.optimize
			self.tensorboard_init()

	def connect_front_back_to_separator(self, separator):
		with self.graph.as_default():
			self.connect_front(separator)
			self.sepNet.prediction
			self.sepNet.output = self.sepNet.separate
			self.separator
			self.back
			self.cost


	##
	## Front End creating STFT like data + P Matrix
	##
	@ops.scope
	def front(self):
		# Front-End

		# x : [ B-tot , 1, L , 1]
		input_front =  tf.expand_dims(tf.expand_dims(self.x, 2), 1)

		self.window_filter = get_scope_variable('window', 'w', shape=[self.window], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		self.bases = get_scope_variable('bases', 'bases', shape=[self.window, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		self.conv_filter = tf.reshape(tf.expand_dims(self.window_filter,1)*self.bases , [1, self.window, 1, self.N])
		
		# 1 Dimensional convolution along T axis with a window length = 1024
		# And N = 256 filters
		self.X = tf.nn.conv2d(input_front, self.conv_filter, strides=[1, 1, 1, 1], padding="SAME", name='Conv_STFT')
		
		self.X = tf.reshape(self.X, [self.B_tot, -1, self.N, 1])

		# self.X_abs = tf.abs(X)
		self.T = tf.shape(self.X)[1]

		# Max Pooling with argmax for unpooling later in the back-end layer
		self.y, argmax = tf.nn.max_pool_with_argmax(self.X, (1, self.max_pool_value, 1, 1),
													strides=[1, self.max_pool_value, 1, 1], padding="SAME")
		
		y_shape = tf.shape(self.y)
		self.p_hat = tf.reduce_sum(tf.abs(self.y), axis=[1,2,3])/(tf.cast(y_shape[1]*y_shape[2], tf.float32))

		return self.y, argmax


	@ops.scope
	def separator(self):
		# shape = [B_tot, T_, N, 1], shape = [B(1+S), T , N, 1], [B(1+S), T_ , N, 1]
		separator_in, argmax_in = self.front

		if self.pretraining:
			self.T_max_pooled = tf.shape(separator_in)[1]
			return tf.reshape(separator_in,[self.B*self.S, self.T_max_pooled, self.N, 1]), argmax_in
		else:
			argmax_in = argmax_in[:self.B, :, :, :]
			argmax_in = tf.tile(argmax_in, [self.S, 1, 1, 1])
			return self.sepNet.output, argmax_in

	@ops.scope
	def back(self):
		# Back-End
		input_tensor, argmax = self.separator
		print input_tensor

		# Unpooling (the previous max pooling)
		self.unpooled = unpool(input_tensor, argmax, ksize=[1, self.max_pool_value, 1, 1], scope='unpool')

		self.recons = self.unpooled

		output = tf.reshape(self.recons, [self.B*self.S, 1, self.T, self.N])

		output = tf.nn.conv2d_transpose(output , filter=self.conv_filter,
									 output_shape=[self.B*self.S, 1, self.L, 1],
									 strides=[1, 1, 1, 1])
		self.out = output
		output = tf.reshape(output, [self.B, self.S, self.L], name='back_output')

		return output


	def logfunc(self, x, x2):
		cx = tf.clip_by_value(x, 1e-10, 1.0)
		cx2 = tf.clip_by_value(x2, 1e-10, 1.0)
		return tf.multiply(x, tf.log(tf.div(cx,cx2)))


	def kl_div(self, p, p_hat):
		inv_p = 1 - p
		inv_p_hat = 1 - p_hat 
		return self.logfunc(p, p_hat) + self.logfunc(inv_p, inv_p_hat)


	@ops.scope
	def cost(self):
		# Definition of cost for Adapt model
		# Regularisation
		# shape = [B_tot, T, N]
		self.sparse_reg = self.beta * tf.reduce_mean(self.kl_div(self.p, self.p_hat))
		
		self.reg = self.l * (tf.nn.l2_loss(self.conv_filter))
		
		# input_shape = [B, S, L]
		# Doing l2 norm on L axis : 
		if self.pretraining:
			self.cost_1 = 0.5 * tf.reduce_sum(tf.pow(self.X_non_mix - self.back, 2), axis=2) / tf.cast(self.L, tf.float32)
			# shape = [B, S]
			# Compute mean over the speakers
			cost = tf.reduce_mean(self.cost_1, 1)
		else:
			# Compute loss over all possible permutations
			
			perms = list(permutations(range(self.S))) # ex with 3: [0, 1, 2], [0, 2 ,1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]
			length_perm = len(perms)
			perms = tf.reshape(tf.constant(perms), [1, length_perm, self.S, 1])
			perms = tf.tile(perms, [self.B, 1, 1, 1])

			batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1, 1]), [1, length_perm, self.S, 1])
			perm_range = tf.tile(tf.reshape(tf.range(length_perm, dtype=tf.int32), shape=[1, length_perm, 1, 1]), [self.B, 1, self.S, 1])
			indicies = tf.concat([batch_range, perm_range, perms], axis=3)

			# [B, P, S, L]
			permuted_back = tf.gather_nd(tf.tile(tf.reshape(self.back, [self.B, 1, self.S, self.L]), [1, length_perm, 1, 1]), indicies) # 

			X_nmr = tf.reshape(self.X_non_mix, [self.B, 1, self.S, self.L])

			cost = 0.5 * tf.reduce_sum(tf.pow(X_nmr - permuted_back, 2), axis=3) / tf.cast(self.L, tf.float32)
			cost = tf.reduce_mean(cost, axis = 2) # Take the mean among speakers
			cost = tf.reduce_min(cost, axis = 1) # Take the permutation minimizing the cost
		
		# shape = [B]
		# Compute mean over batches
		self.MSE = tf.reduce_mean(cost, 0) 
		self.cost = self.MSE #+ self.sparse_reg + self.reg

		return self.cost

	def select_optimizer(self,string):
		return {
			'Adam': tf.train.AdamOptimizer,
			'RMS': tf.train.RMSPropOptimizer,
		}[string]

	@ops.scope
	def optimize(self):
		if hasattr(self, 'trainable_variables') == False:
			self.trainable_variables = tf.global_variables()	

		# optimizer = self.select_optimizer(self.optimizer)(self.learning_rate)
		# gradients, variables = zip(*optimizer.compute_gradients(self.cost, var_list=self.trainable_variables))
		# # gradients, _ = tf.clip_by_global_norm(gradients, 200.0)
		# optimize = optimizer.apply_gradients(zip(gradients, variables))
		optimize = self.select_optimizer(self.optimizer)(self.learning_rate).minimize(self.cost, var_list=self.trainable_variables)
		return optimize

	def save(self, step):
		self.saver.save(self.sess, os.path.join(config.log_dir, self.folder ,self.runID, "model.ckpt"))  # , step)


	def train(self, X_mix, X_non_mix, learning_rate, step, ind_train=None):
		if ind_train is None:
			summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_non_mix, self.training:True, self.learning_rate:learning_rate})
		else:
			summary, _, cost = self.sess.run([self.merged, self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_non_mix, self.training:True, self.Ind:ind_train, self.learning_rate:learning_rate})
		
		self.train_writer.add_summary(summary, step)
		return cost

	def pretrain(self, X_non_mix, learning_rate, step):
		# options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		# run_metadata = tf.RunMetadata()

		_ ,summary, cost = self.sess.run([self.optimize, self.merged, self.cost], {self.X_non_mix:X_non_mix, self.training:True, self.learning_rate:learning_rate})#,  options=options, run_metadata=run_metadata)
		self.train_writer.add_summary(summary, step)

		# fetched_timeline = timeline.Timeline(run_metadata.step_stats)
		# chrome_trace = fetched_timeline.generate_chrome_trace_format()
		# with open('timeline_01.json', 'w') as f:
		# 	f.write(chrome_trace)
		return cost

	def train_no_sum(self, X_mix, X_in, learning_rate, step, ind_train=None):
		if ind_train is None:
			_, cost = self.sess.run([self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in, self.training:True, self.learning_rate:learning_rate})
		else:
			_, cost = self.sess.run([self.optimize, self.cost], {self.X_mix: X_mix, self.X_non_mix:X_in, self.training:True, self.Ind:ind_train, self.learning_rate:learning_rate})
		return cost

	def connect_front(self, separator_class):
		# Separate Mixed and Non Mixed 'spectrograms'
		with tf.name_scope('split_front'):

			X = tf.reshape(self.front[0][:self.B, :, :], [self.B, -1, self.N])

			X_non_mix = tf.cond(self.training, lambda: tf.transpose(tf.reshape(self.front[0][self.B:, :, :, :], [self.B, self.S, -1, self.N]), [0,2,3,1]), lambda: X)

			input = tf.cond(self.training, lambda: (X, X_non_mix), lambda: (X, X),strict=False)
		self.sepNet = separator_class(input, self)

	def connect_back(self):
		self.back

	def freeze_front(self):
		training_var = tf.trainable_variables()
		training_var.remove(self.bases)
		training_var.remove(self.window_filter)
		self.trainable_variables = training_var
