# -*- coding: utf-8 -*-
from utils.ops import unpool, variable_summaries, get_scope_variable, scope, kl_div
import os
import config
import tensorflow as tf
from itertools import combinations, permutations
from tensorflow.python.saved_model import builder as saved_model_builder
import numpy as np
from network import Network

#############################################
#     Adaptive Front and Back End Model     #
#############################################
		
class Adapt(Network):
	def __init__(self, *args, **kwargs):
		##
		## Model Configuration 
		##
		super(Adapt, self).__init__(*args, **kwargs)

		if kwargs is not None:
			self.N = kwargs['filters']
			self.max_pool_value = kwargs['max_pool']
			self.l = kwargs['regularization']
			self.beta = kwargs['beta']
			self.p = kwargs['sparsity']
			self.window = kwargs['window_size']
			self.pretraining = kwargs['pretraining']
			self.overlap_coef = kwargs['overlap_coef']
			self.overlap_value = kwargs['overlap_value']
			self.loss = kwargs['loss']
			self.separation = kwargs['separation']
			self.with_max_pool = kwargs['with_max_pool']
			self.with_average_pool = kwargs['with_average_pool']
			self.hop_size = kwargs['hop_size']

		with tf.get_default_graph().as_default():

			with tf.name_scope('preprocessing'):
				if self.pretraining:
					self.x = tf.concat([self.x_mix,  tf.reshape(self.x_non_mix, [self.B*self.S, self.L])], axis=0)
					self.B_tot = tf.shape(self.x)[0]
				else:

					self.x =tf.concat([self.x_mix,  tf.reshape(self.x_non_mix, [self.B*self.S, self.L])], axis=0)
					self.B_tot = self.B*(self.S+1)
					
			if self.pretraining:
				self.front
				self.separator
				self.back
				self.cost_model = self.cost
				self.finish_construction()
				self.optimize
			else:
				# self.sepNet = kwargs['separator']
				self.front

	def create_centroids_saver(self):
		self.centroids_saver = tf.train.Saver([self.sepNet.speaker_vectors], max_to_keep=10000000)

	def savedModel(self):
		path = os.path.join(config.log_dir,self.folder ,self.runID, 'SavedModel')
		builder = saved_model_builder.SavedModelBuilder(path)

		# Build signatures
		input_tensor_info = tf.saved_model.utils.build_tensor_info(self.x_mix)
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

	##
	## Front End creating STFT like data
	##
	@scope
	def front(self):
		# Front-End

		# x : [ Btot , 1, L , 1]
		# Equivalent to B_tot batches of image of height = 1, width = L and 1 channel -> for Conv1D with Conv2D
		input_front =  tf.reshape(self.x, [self.B_tot, 1, self.L, 1])

		# Filter [filter_height, filter_width, input_channels, output_channels] = [1, W, 1, N]
		self.window_filter = get_scope_variable('window', 'w', shape=[self.window], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		self.bases = get_scope_variable('bases', 'bases', shape=[self.window, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		self.conv_filter = tf.reshape(tf.abs(tf.expand_dims(self.window_filter,1))*self.bases , [1, self.window, 1, self.N])
		# self.conv_filter = get_scope_variable('filters_front','filters_front', shape=[1, self.window, 1, self.N])
		variable_summaries(self.conv_filter)

		# 1 Dimensional convolution along T axis with a window length = self.window
		# And N = 256 filters -> Create a [Btot, 1, T, N]
		self.T = tf.shape(input_front)[2]

		if self.with_max_pool:
			self.X = tf.nn.conv2d(input_front, self.conv_filter, strides=[1, 1, 1, 1], padding="SAME", name='Conv_STFT')
			self.y, self.argmax = tf.nn.max_pool_with_argmax(self.X, [1, 1, self.max_pool_value, 1], 
										strides=[1, 1, self.max_pool_value, 1], padding="VALID", name='output')
			print self.argmax
		elif self.with_average_pool:
			self.X = tf.nn.conv2d(input_front, self.conv_filter, strides=[1, 1, 1, 1], padding="SAME", name='Conv_STFT')
			self.y = tf.layers.average_pooling2d(self.X, (1, self.max_pool_value), strides=(1, self.hop_size), name='output')		
		else:
			self.y = tf.nn.conv2d(input_front, self.conv_filter, strides=[1, 1, self.hop_size, 1], padding="SAME", name='Conv_STFT')

		# Reshape to Btot batches of T x N images with 1 channel
		# [Btot, 1, T_pool, N] -> [Btot, T-pool, N, 1]
		self.y = tf.transpose(self.y,[0, 2, 3, 1], name='output')

		tf.summary.image('front/output', self.y, max_outputs=3)
		y_shape = tf.shape(self.y)
		y = tf.reshape(self.y, [self.B_tot, y_shape[1]*y_shape[2]])
		self.p_hat = tf.reduce_sum(tf.abs(y), 0)
		self.sparse_constraint = tf.reduce_sum(kl_div(self.p, self.p_hat))

		return self.y

	@scope
	def separator(self):
		# shape = [B_tot, T_, N, 1], shape = [B(1+S), T , N, 1], [B(1+S), T_ , N, 1]
		separator_in = self.front
		# Compute the overlapping rate:
		nb = 2
		comb = list(combinations(range(self.S), nb))
		len_comb = len(comb)
		combs = tf.reshape(tf.constant(comb), [1, len_comb, 2, 1])
		combs = tf.tile(combs, [self.B, 1, 1, 1])
		batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1, 1]), [1, len_comb, nb, 1])
		comb_range = tf.tile(tf.reshape(tf.range(len_comb, dtype=tf.int32), shape=[1, len_comb, 1, 1]), [self.B, 1, nb, 1])
		indicies = tf.concat([batch_range, comb_range, combs], axis=3)
		self.T_max_pooled = tf.shape(separator_in)[1]
		input = tf.reshape(separator_in, [self.B_tot, self.T_max_pooled, self.N])
		input_non_mix = tf.reshape(input[self.B:, : , :], [self.B, self.S, self.T_max_pooled, self.N]) # B*S others non mix
		comb_non_mix = tf.gather_nd(tf.tile(tf.reshape(input_non_mix, [self.B, 1, self.S, self.T_max_pooled*self.N]), [1, len_comb, 1, 1]), indicies) # 

		# Combination of non mixed representations : [B, len(comb), nb, T*N]
		comb_non_mix = tf.abs(comb_non_mix)
		measure = 1.0 - tf.abs(comb_non_mix[:,:,0,:] - comb_non_mix[:,:,1,:]) / (tf.reduce_max(comb_non_mix, axis=2) + 1e-8)
		overlapping = tf.reduce_mean(measure, -1) # Mean over the bins
		overlapping = tf.reduce_mean(overlapping, -1) # Mean over combinations
		self.overlapping = tf.reduce_mean(overlapping, -1) # Mean over batces
		self.overlapping_constraint = self.overlapping

		input_mix = tf.reshape(input[:self.B, : , :], [self.B, 1, self.T_max_pooled, self.N]) # B first batches correspond to mix input
		input_non_mix = tf.reshape(input[self.B:, : , :], [self.B, self.S, self.T_max_pooled, self.N]) # B*S others non mix

		input_mix = tf.tile(input_mix, [1, self.S, 1, 1])

		filters = tf.divide(input_non_mix, input_mix)
		output = tf.reshape(input_mix * filters, [self.B*self.S, self.T_max_pooled, self.N, 1])
		output = tf.transpose(output, [0, 3, 1, 2])

		self.filters = filters

		if self.pretraining:
			self.T_max_pooled = tf.shape(separator_in)[1]

			input = tf.reshape(separator_in, [self.B_tot, self.T_max_pooled, self.N])

			
			if self.separation == 'mask':
				filters = tf.divide(input_non_mix, input_mix)
				output = tf.reshape(input_mix * filters, [self.B*self.S, self.T_max_pooled, self.N, 1])
				output = tf.transpose(output, [0, 3, 1, 2])

				self.filters = filters

			elif self.separation == 'perfect':
				# From [a, b, c ,d] -> [a+b+c+d, a+b+c+d, a+b+c+d, a+b+c+d]
				tiled_sum = tf.tile(tf.reduce_sum(input_non_mix, 1, keep_dims=True), [1, self.S, 1, 1])
				# From [a+b+c+d, a+b+c+d, a+b+c+d, a+b+c+d] ->  [b+c+d, a+c+d, a+b+d, a+b+c]
				X_add = tiled_sum - input_non_mix
				output = input_mix - X_add

				# with tf.control_dependencies([tf.assert_equal(output,input_non_mix)]):
				output = tf.reshape(output, [self.B*self.S, self.T_max_pooled, self.N, 1])

			return output
		else:
			output = self.sepNet.output
			self.prediction = self.sepNet.prediction
			self.true_masks = self.sepNet.true_masks
			self.X_non_mix = self.sepNet.X_non_mix
			self.X = self.sepNet.X
			return output

	@scope
	def back(self):

		input_tensor = self.separator

		if self.with_max_pool:

			argmax_mix = self.argmax[:self.B]
	 
			repeats = [self.S, 1, 1 ,1]
			shape = tf.shape(argmax_mix)
			argmax_mix = tf.expand_dims(argmax_mix, 1)
			argmax_mix = tf.tile(argmax_mix, [1, self.S, 1, 1 ,1])
			argmax_mix = tf.reshape(argmax_mix, shape*repeats)

			output_shape = [self.B*self.S, 1, self.T, self.N]

			output = unpool(input_tensor, argmax_mix, ksize=[1, 1, self.max_pool_value, 1], output_shape=output_shape, scope='unpool')
			output = tf.reshape(output, [self.B*self.S, 1, self.T, self.N])

		elif self.with_average_pool:
			input_tensor = tf.reshape(input_tensor, [self.B*self.S, 1, self.T_max_pooled, self.N])
			output = tf.keras.layers.UpSampling2D((1, self.max_pool_value))(input_tensor)
			output = tf.reshape(output, [self.B*self.S, 1, self.T, self.N])
		else:
			output = tf.reshape(input_tensor, [self.B*self.S, 1, self.T_max_pooled, self.N])

		self.window_filter_2 = get_scope_variable('window', 'value', shape=[self.window], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		self.bases_2 = get_scope_variable('bases', 'value', shape=[self.window, self.N], initializer=tf.contrib.layers.xavier_initializer_conv2d())
		self.conv_filter_2 = tf.reshape(tf.abs(tf.expand_dims(self.window_filter_2,1))*self.bases_2 , [1, self.window, 1, self.N], name='filters')# self.conv_filter_2 = tf.Variable(self.conv_filter.initialized_value(), name="filters_back")

		if not (self.with_max_pool or self.with_average_pool):
			strides = [1, 1, self.max_pool_value, 1]
		else:
			strides = [1, 1, 1, 1]

		output = tf.nn.conv2d_transpose(output , filter=self.conv_filter_2,
									 output_shape=[self.B*self.S, 1, self.L, 1],
									 strides=strides, padding='SAME')

		output = tf.reshape(output, [self.B, self.S, self.L], name='back_output')
		self.output = output

		sdr_improvement, sdr = self.sdr_improvement(self.x_non_mix, output)
		self.sdr_imp = sdr_improvement
		return output

	@scope
	def cost(self):
		# Definition of cost for Adapt model
		# Regularisation
		# shape = [B_tot, T, N]		
		# regularization = tf.nn.l2_loss(self.conv_filter_2) + tf.nn.l2_loss(self.conv_filter)
		
		# non_negativity_1 = tf.reduce_sum(tf.square(tf.where(self.conv_filter < 0, self.conv_filter, tf.zeros_like(self.conv_filter))))
		# non_negativity_2 = tf.reduce_sum(tf.square(tf.where(self.conv_filter_2 < 0, self.conv_filter_2, tf.zeros_like(self.conv_filter_2))))
		# nn = 0.5 * (non_negativity_1 + non_negativity_2)
		neg = tf.square(tf.where(self.front < 0, self.front, tf.zeros_like(self.front)))
		nn = 10*tf.reduce_mean(tf.reduce_sum(neg, [1,2,3]))

		regularization = self.l * (tf.nn.l2_loss(self.conv_filter_2)+ tf.nn.l2_loss(self.conv_filter))


		# input_shape = [B, S, L]
		# Doing l2 norm on L axis : 
		if self.pretraining:
			
			l2 = tf.reduce_sum(tf.square(self.x_non_mix - self.back), axis=-1)
			l2 = tf.reduce_sum(l2, -1) # Sum over all the speakers 
			l2 = tf.reduce_mean(l2, -1) # Mean over batches

			sdr_improvement, sdr = self.sdr_improvement(self.x_non_mix, self.back)
			self.sdr_imp = sdr_improvement
			sdr = tf.reduce_mean(sdr) # Mean over speakers
			sdr = tf.reduce_mean(sdr) # Mean over batches

			if self.loss == 'l2':
				loss = l2
			elif self.loss == 'sdr':
				loss = sdr
			else:
				loss = l2 + sdr

			# loss = loss + nn

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

			X_nmr = tf.reshape(self.x_non_mix, [self.B, 1, self.S, self.L])

			l2 = tf.reduce_sum(tf.square(X_nmr - permuted_back), axis=-1) # L2^2 norm
			l2 = tf.reduce_min(l2, axis=1) # Get the minimum over all possible permutations : B S
			l2 = tf.reduce_sum(l2, -1)
			l2 = tf.reduce_mean(l2, -1)

			sdr_improvement, sdr = self.sdr_improvement(X_nmr, self.back, True)
			self.sdr_imp = sdr_improvement
			sdr = tf.reduce_min(sdr, 1) # Get the minimum over all possible permutations : B S
			sdr = tf.reduce_sum(sdr, -1)
			sdr = tf.reduce_mean(sdr, -1)

			if self.loss == 'l2':
				loss = l2
			elif self.loss == 'sdr':
				loss = sdr
			else:
				loss = 1e-3*l2 + sdr

		# shape = [B]
		# Compute mean over batches
		cost_value = loss
		if self.beta != 0.0:
			cost_value += self.beta * self.sparse_constraint
		if self.l != 0.0:
			cost_value += self.l * regularization 
		if self.overlap_coef != 0.0:
			cost_value += self.overlap_coef * self.overlapping_constraint

		variable_summaries(self.conv_filter_2)

		tf.summary.audio(name= "audio/output/reconstructed", tensor = tf.reshape(self.back, [-1, self.L]), sample_rate = config.fs, max_outputs=2)
		
		with tf.name_scope('loss_values'):
			tf.summary.scalar('l2_loss', l2)
			tf.summary.scalar('SDR', sdr)
			tf.summary.scalar('SDR_improvement', sdr_improvement)			
			tf.summary.scalar('Non negativity loss', nn)			
			tf.summary.scalar('sparsity', tf.reduce_mean(self.p_hat))
			tf.summary.scalar('sparsity_loss', self.beta * self.sparse_constraint)
			tf.summary.scalar('L2_reg', self.l * regularization)
			tf.summary.scalar('loss', cost_value)
			tf.summary.scalar('overlapping', self.overlapping)
			tf.summary.scalar('overlapping_loss', self.overlap_coef * self.overlapping_constraint)

		return cost_value

	def test_prediction(self, X_mix_test, X_non_mix_test, step):
		pred, y = self.sess.run([self.sepNet.prediction, self.sepNet.y_test_export], {self.x_mix: X_mix_test, self.x_non_mix:X_non_mix_test, self.training:True})
		pred = np.reshape(pred, [X_mix_test.shape[0], -1, 40])
		labels = [['r' if b == 1 else 'b' for b in batch]for batch in y]
		np.save(os.path.join(config.log_dir, self.folder ,self.runID, "bins-{}".format(step)), pred)
		np.save(os.path.join(config.log_dir, self.folder ,self.runID, "labels-{}".format(step)), labels)

	def connect_front(self, separator_class):
		self.sepNet = separator_class(True, **self.args)

	def connect_only_front_to_separator(self, separator, freeze_front=True):
		self.connect_front(separator)
		self.sepNet.output = self.sepNet.prediction
		self.cost_model = self.sepNet.cost
		self.back # To save the back values !
		var_list =[v for v in tf.global_variables() if ('back/' in v.name or 'front/' in v.name)]
		self.create_saver(subset=var_list)
		self.restore_model(self.args['model_folder'])
		self.finish_construction()
		self.freeze_all_with('front/')
		self.freeze_all_with('back/')
		self.optimize
		self.tensorboard_init()

	def connect_enhance_to_separator(self, separator):
		self.connect_front(separator)
		self.sepNet.output = self.sepNet.enhance
		self.cost_model = self.sepNet.enhance_cost
		self.back # To save the back values !
		var_list =[v for v in tf.global_variables() 
			if ('back/' in v.name or 'front/' in v.name or 'prediction/' in v.name or 'speaker_centroids' in v.name )]
		self.create_saver(subset=var_list)
		self.restore_model(self.args['model_folder'])
		self.finish_construction()
		self.freeze_all_except('enhance/')
		self.optimize
		self.tensorboard_init()

	def restore_front_separator(self, sess, path, separator):
		self.connect_front(separator)
		self.sepNet.output = self.sepNet.prediction
		self.back
		self.restore_model(sess, path)