# -*- coding: utf-8 -*-
from utils.ops import scope, AMSGrad, BLSTM, f_props, log10, Conv1D
import os
import config
import tensorflow as tf
import haikunator
from itertools import compress, permutations
import numpy as np
import json
from tensorflow.python.framework import ops

class Network(object):

	"""docstring for Network"""
	def __init__(self, graph=None, *args, **kwargs):		
		# Constant seed for uniform results
		tf.set_random_seed(42)
		np.random.seed(42)

		##
		## Model Configuration 
		if kwargs is not None:
			self.folder = kwargs['type']
			self.S = kwargs['nb_speakers']
			self.args = kwargs
			self.learning_rate = kwargs['learning_rate']
			self.my_opt = kwargs['optimizer']
			self.decay_epoch = kwargs['decay_epoch']
			self.gradient_clip = kwargs['gradient_norm_clip']
		else:
			raise Exception('Keyword Arguments missing ! Please add the right arguments in input | check doc')

		if graph is None:
			# Run ID
			self.runID = haikunator.Haikunator().haikunate()
			print 'ID : {}'.format(self.runID)

			if not kwargs['pipeline']:
				#Create a graph for this model
				self.graph = tf.Graph()

				with self.graph.as_default():

					with tf.name_scope('inputs'):

						# Boolean placeholder signaling if the model is in learning/training mode
						self.training = tf.placeholder(tf.bool, name='is_training')

						# Batch of raw non-mixed audio
						# shape = [ batch size , number of speakers, samples ] = [ B, S, L]
						self.x_non_mix = tf.placeholder("float", [None, None, None], name='non_mix_input')

						# Batch of raw mixed audio - Input data
						# shape = [ batch size , samples ] = [ B , L ]
						self.x_mix = tf.placeholder("float", [None, None], name='mix_input')

						# Speakers indicies used in the mixtures
						# shape = [ batch size, #speakers]
						self.I = tf.placeholder(tf.int32, [None, None], name='indicies')

						shape_in = tf.shape(self.x_mix)
						self.B = shape_in[0]
						self.L = shape_in[1]
			else:
				with tf.get_default_graph().as_default():
					with tf.name_scope('inputs'):

						# Boolean placeholder signaling if the model is in learning/training mode
						self.training = tf.placeholder(tf.bool, name='is_training')

						# Batch of raw non-mixed audio
						# shape = [ batch size , number of speakers, samples ] = [ B, S, L]
						self.x_non_mix = tf.identity(kwargs['non_mix'], name='non_mix_input')

						# Batch of raw mixed audio - Input data
						# shape = [ batch size , samples ] = [ B , L ]
						self.x_mix =  tf.identity(kwargs['mix'], name='mix_input')

						# Speakers indicies used in the mixtures
						# shape = [ batch size, #speakers]
						self.I =  tf.identity(kwargs['ind'], name='indicies')

						shape_in = tf.shape(self.x_mix)
						self.B = shape_in[0]
						self.L = shape_in[1]

						tf.summary.audio(name= "audio/input/non-mixed", tensor = tf.reshape(self.x_non_mix, [-1, self.L]), sample_rate = config.fs, max_outputs=2)
						tf.summary.audio(name= "audio/input/mixed", tensor = self.x_mix[:self.B], sample_rate = config.fs, max_outputs=1)

	def tensorboard_init(self):
		self.saver = tf.train.Saver()

		sums = ops.get_collection(ops.GraphKeys.SUMMARIES)
		train_keys_summaries = []
		valid_keys_summaries = []
		test_keys_summaries = []

		for s in sums:
			if not ('input' in s.name or 'output' in s.name):
				train_keys_summaries.append(s)
			else:
				valid_keys_summaries.append(s)
			if 'SDR_improvement' in s.name:
				valid_keys_summaries.append(s)
				test_keys_summaries.append(s)
			if 'audio' in s.name or 'stft' in s.name or 'mask' in s.name:
				test_keys_summaries.append(s)

		self.merged_train = tf.summary.merge(train_keys_summaries)
		if len(valid_keys_summaries) != 0:
			self.merged_valid = tf.summary.merge(valid_keys_summaries)
		else:
			self.merged_valid = None

		if len(test_keys_summaries) != 0:
			self.merged_test = tf.summary.merge(test_keys_summaries)
		else:
			self.merged_test = None

		self.train_writer = tf.summary.FileWriter(os.path.join(config.log_dir,self.folder,self.runID,'train'), tf.get_default_graph())
		self.valid_writer = tf.summary.FileWriter(os.path.join(config.log_dir,self.folder,self.runID,'valid'))
		self.test_writer = tf.summary.FileWriter(os.path.join(config.log_dir,self.folder,self.runID,'test'))

		# Save arguments
		with open(os.path.join(config.log_dir,self.folder,self.runID,'params'), 'w') as f:
			del self.args['mix']
			del self.args['non_mix']
			del self.args['ind']
			json.dump(self.args, f)

	def create_saver(self, subset=None):
		if subset is None:
			self.saver = tf.train.Saver()
		else:
			self.saver = tf.train.Saver(subset)

	# Restore last checkpoint of the current graph using the total path
	# This method is used when we plug a new layer
	def restore_model(self, path):
		self.saver.restore(tf.get_default_session(), tf.train.latest_checkpoint(path))
	
	# Restore the last checkpoint of the current trained model
	# This function is maintly used during the test phase
	def restore_last_checkpoint(self):
		self.saver.restore(tf.get_default_session(), tf.train.latest_checkpoint(os.path.join(config.log_dir, self.folder ,self.runID)))

	def init_all(self):
		tf.get_default_session().run(tf.global_variables_initializer())
 
	def non_initialized_variables(self):
		global_vars = tf.global_variables()
		is_not_initialized = tf.get_default_session().run([~(tf.is_variable_initialized(var)) \
									   for var in global_vars])
		not_initialized_vars = list(compress(global_vars, is_not_initialized))
		
		print 'not init: ', [v.name for v in not_initialized_vars]

		if len(not_initialized_vars):
			init = tf.variables_initializer(not_initialized_vars)
			return init

	def initialize_non_init(self):
		non_init = self.non_initialized_variables()
		if non_init is not None:
			tf.get_default_session().run(non_init)

	@scope
	def optimize(self):
		print 'Train the following variables :', map(lambda x: x.name, self.trainable_variables)
		
		global_epoch = tf.Variable(0, name="global_epoch", trainable=False)

		self.increment_epoch = tf.assign(global_epoch, global_epoch+1)

		learning_rate = tf.train.exponential_decay(self.learning_rate, 
				global_epoch, self.decay_epoch, 
				0.5, staircase=True)

		tf.summary.scalar('learning_rate', learning_rate)

		if self.my_opt == 'Adam':
			optimizer = AMSGrad(self.learning_rate, epsilon=0.001)
		elif self.my_opt == 'SGD':
			optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
		elif self.my_opt == 'RMSProp':
			optimizer = tf.train.RMSPropOptimizer(learning_rate)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			gradients, variables = zip(*optimizer.compute_gradients(self.cost_model, var_list=self.trainable_variables))
			if self.gradient_clip != 0.:
				gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
			optimize = optimizer.apply_gradients(zip(gradients, variables))
			return optimize

	def sdr_improvement(self, s_target, s_approx, with_perm=False):
		# B S L or B P S L
		mix = tf.tile(tf.expand_dims(self.x_mix, 1) ,[1, self.S, 1])

		s_target_norm = tf.reduce_sum(tf.square(s_target), axis=-1)
		s_approx_norm = tf.reduce_sum(tf.square(s_approx), axis=-1)
		mix_norm = tf.reduce_sum(tf.square(mix), axis=-1)

		s_target_s_2 = tf.square(tf.reduce_sum(s_target*s_approx, axis=-1))
		s_target_mix_2 = tf.square(tf.reduce_sum(s_target*mix, axis=-1))

		sep = 1.0/((s_target_norm*s_approx_norm)/s_target_s_2 - 1.0)
		separated = 10. * log10(sep)
		non_separated = 10. * log10(1.0/((s_target_norm*mix_norm)/s_target_mix_2 - 1.0))

		loss = (s_target_norm*s_approx_norm)/s_target_s_2

		val = separated - non_separated
		val = tf.reduce_mean(val , -1) # Mean over speakers
		if not with_perm:
			val = tf.reduce_mean(val , -1) # Mean over batches
		else:
			val = tf.reduce_mean(val , 0) # Mean over batches
			val = tf.reduce_min(val, -1)

		return val, loss

	def save(self, step):
		path = os.path.join(config.log_dir, self.folder ,self.runID, "model")
		self.saver.save(tf.get_default_session(), path, step)
		return path

	def train(self, feed_dict, step):
		feed_dict.update({self.training:True})
		summary, _, cost = tf.get_default_session().run([self.merged_train, self.optimize, self.cost_model], feed_dict)
		self.train_writer.add_summary(summary, step)
		return cost

	def infer(self, feed_dict, step):
		feed_dict.update({self.training:False})
		output = tf.get_default_session().run([self.x_mix, self.x_non_mix, self.output], feed_dict)
		return output

	def improvement(self, feed_dict, step):
		feed_dict.update({self.training:False})
		output = tf.get_default_session().run([self.x_mix, self.x_non_mix, self.sdr_imp], feed_dict)
		return output

	def valid_batch(self, feed_dict, step):
		sess = tf.get_default_session()
		feed_dict.update({self.training:False})
		if self.merged_valid is None:
			cost = sess.run(self.cost_model, feed_dict)
		else:
			cost, summary = sess.run([self.cost_model, self.merged_valid], feed_dict)
			self.valid_writer.add_summary(summary, step)
		return cost

	def get_embeddings(self, feed_dict):
		sess = tf.get_default_session()
		feed_dict.update({self.training:False})
		return sess.run(self.prediction, feed_dict)

	def test_batch(self, feed_dict):
		sess = tf.get_default_session()
		feed_dict.update({self.training:False})
		return sess.run(self.cost_model, feed_dict)

	def test(self, feed_dict):
		sess = tf.get_default_session()
		feed_dict.update({self.training:True})
		return sess.run(self.y, feed_dict)

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

	def freeze_all_except(self, *prefix):
		to_train = []
		for var in self.trainable_variables:
			for p in prefix:
				if p in var.name:
					to_train.append(var)
					break
		self.trainable_variables = to_train

	@classmethod
	def load(cls, path, modified_args):
		# Load parameters used for the desired model to load
		params_path = os.path.join(path, 'params')
		with open(params_path) as f:
			args = json.load(f)
			keys_to_update = ['learning_rate','epochs','batch_size', 'chunk_size', 'nb_speakers',
			'regularization','overlap_coef','loss','beta','model_folder', 'type','pretraining', 'with_silence',
			'beta_kmeans', 'nb_tries', 'nb_steps', 'threshold', 'optimizer']
			to_modify = { key: modified_args[key] for key in keys_to_update if key in modified_args.keys() }
			to_modify.update({key: val for key, val in modified_args.items() if key not in args.keys()})
		# Update with new args such as 'pretraining' or 'type'
		args.update(to_modify)
		print "LOADED = ", args
		# Create a new Adapt model with these parameters
		return cls(**args)

	def finish_construction(self):
		self.trainable_variables = tf.global_variables()

from models.Kmeans_2 import KMeans

class Separator(Network):

	def __init__(self, plugged=False, *args, **kwargs):
		super(Separator, self).__init__(plugged, *args, **kwargs)

		self.num_speakers = kwargs['tot_speakers']
		self.layer_size = kwargs['layer_size']
		self.embedding_size = kwargs['embedding_size']
		self.normalize = kwargs['no_normalize']
		self.nb_layers = kwargs['nb_layers']
		self.a = kwargs['mask_a']
		self.b = kwargs['mask_b']
		self.rdropout = kwargs['recurrent_dropout']

		#Preproc
		self.normalize_input = kwargs['normalize_separator']
		self.abs_input = kwargs['abs_input']
		self.pre_func = kwargs['pre_func']
		self.silent_threshold = kwargs['silence_mask_db']

		# Loss Parameters
		self.loss_with_silence = kwargs['silence_loss']
		self.threshold_silence_loss = kwargs['threshold_silence_loss']
		self.function_mask = kwargs['function_mask']

		# Kmeans Parameters
		self.beta = kwargs['beta_kmeans']
		self.threshold = kwargs['threshold']
		self.with_silence = kwargs['with_silence']
		self.nb_tries = kwargs['nb_tries']
		self.nb_steps = kwargs['nb_steps']

		# Negative Sampling for L41
		self.sampling = kwargs['sampling']
		self.ns_rate = kwargs['ns_rate']
		self.ns_method = kwargs['ns_method']

		# Adding architectures
		self.add_dilated = kwargs['add_dilated']

		self.graph = tf.get_default_graph()

		self.plugged = plugged
		# If the Separator is not independant but using a front layer
		if self.plugged:
			self.F = kwargs['filters']

			with self.graph.as_default():

				self.training = self.graph.get_tensor_by_name('inputs/is_training:0')

				front = self.graph.get_tensor_by_name('front/output:0')

				self.B = tf.shape(self.graph.get_tensor_by_name('inputs/non_mix_input:0'))[0]

				with tf.name_scope('split_front'):
					self.X = tf.reshape(front[:self.B, :, :, :], [self.B, -1, self.F]) # Mix input [B, T, N]
					# Non mix input [B, T, N, S]
					self.X_input = tf.identity(self.X)
					self.X_non_mix = tf.transpose(tf.reshape(front[self.B:, :, :, :], [self.B, self.S, -1, self.F]), [0,2,3,1])

				with tf.name_scope('create_masks'):
					# # Batch of Masks (bins label)
					# # shape = [ batch size, T, F, S]
					argmax = tf.argmax(tf.abs(self.X_non_mix), axis=3)
					self.y = tf.one_hot(argmax, self.S, self.a, self.b)
					self.y_test_export = tf.reshape(self.y[:, :, :, 0], [self.B, -1])

					if self.function_mask == 'linear':
						max_ = tf.reduce_max(tf.abs(self.X), [1, 2], keep_dims=True)
						self.y = self.y * tf.expand_dims(tf.abs(self.X)/max_, 3)
					elif self.function_mask == 'sqrt':
						max_ = tf.reduce_max(tf.abs(self.X), [1, 2], keep_dims=True)
						self.y = self.y * tf.expand_dims(tf.sqrt(tf.abs(self.X)/max_), 3)
					elif self.function_mask == 'square':
						max_ = tf.reduce_max(tf.abs(self.X), [1, 2], keep_dims=True)
						self.y = self.y * tf.expand_dims(tf.square(tf.abs(self.X)/max_), 3)

					if self.loss_with_silence:
						max_ = tf.reduce_max(tf.abs(self.X), [1, 2], keep_dims=True)
						log_compare = log10(tf.divide(max_, tf.abs(self.X)))
						mask = tf.cast(log_compare < self.threshold_silence_loss, tf.float32)
						tf.summary.image('separator/silence_mask', tf.expand_dims(mask,3), max_outputs=1)
						self.y = self.y * tf.expand_dims(mask, 3)

				# Speakers indices used in the mixtures
				# shape = [ batch size, #speakers]
				self.I = tf.get_default_graph().get_tensor_by_name('inputs/indicies:0')
		else:
			# STFT hyperparams
			self.window_size = kwargs['window_size']
			self.hop_size = kwargs['hop_size']

			# Network hyperparams
			self.F = kwargs['window_size']//2 +1

	def init_separator(self):
		if self.plugged:

			if self.abs_input:
				self.X = tf.abs(self.X)

			if self.normalize_input == '01':
				self.normalization01
			elif self.normalize_input == 'meanstd':
				self.normalization_mean_std

			self.prediction

		else:
			
			# STFT 
			self.preprocessing

			# Apply a certain function
			if self.pre_func == 'sqrt':
				self.X = tf.sqrt(self.X)
			elif self.pre_func == 'log':
				self.X = log10(self.X + 1e-12)

			# Apply normalization
			if self.normalize_input == '01':
				self.normalization01
			elif self.normalize_input == 'meanstd':
				self.normalization_mean_std

			# Apply silent mask
			if self.silent_threshold > 0:
				max_ = tf.reduce_max(self.X, [1,2], keep_dims=True)
				mask = tf.cast(tf.less(max_ - self.X , self.silent_threshold/20.), tf.float32)
				self.X = mask * self.X

			if self.add_dilated:
				self.dilated

			self.prediction
			#TODO TO IMPROVE !
			if self.args['model_folder'] is None:
			# if 'inference' not in self.folder and 'enhance' not in self.folder and 'finetuning' not in self.folder:
				self.cost_model = self.cost
				self.finish_construction()
				self.optimize

	def add_enhance_layer(self):
		self.separate
		self.enhance
		self.cost_model = self.enhance_cost
		self.finish_construction()
		self.freeze_all_except('enhance')
		self.optimize

	def add_finetuning(self):
		self.separate
		self.postprocessing
		self.cost_finetuning
		self.cost_model = self.cost_finetuning
		self.finish_construction()
		self.freeze_all_except('prediction')
		self.tensorboard_init()
		self.optimize

	@scope
	def preprocessing(self):
		self.stfts = tf.contrib.signal.stft(self.x_mix, 
			frame_length=self.window_size, 
			frame_step=self.hop_size,
			fft_length=self.window_size)

		self.B = tf.shape(self.x_non_mix)[0]

		self.stfts_non_mix = tf.contrib.signal.stft(tf.reshape(self.x_non_mix, [self.B*self.S, -1]), 
			frame_length=self.window_size, 
			frame_step=self.hop_size,
			fft_length=self.window_size)

		tf.summary.image('stft/non_mix', tf.abs(tf.expand_dims(self.stfts_non_mix,3)), max_outputs=3)
		tf.summary.image('stft/mix', tf.abs(tf.expand_dims(self.stfts,3)), max_outputs=3)

		self.X = tf.abs(self.stfts)
		self.X_input = tf.identity(self.X)
		self.X_non_mix =  tf.transpose(tf.abs(tf.reshape(self.stfts_non_mix, [self.B, self.S, -1, self.F])), [0,2,3,1])

		argmax = tf.argmax(self.X_non_mix, axis=3)
		self.y = tf.one_hot(argmax, self.S, self.a, self.b)

	@scope
	def normalization01(self):
		self.min_ = tf.reduce_min(self.X, axis=[1,2], keep_dims=True)
		self.max_ = tf.reduce_max(self.X, axis=[1,2], keep_dims=True)
		self.X = (self.X - self.min_) / (self.max_ - self.min_)

	@scope
	def normalization_m1_1(self):
		self.min_ = tf.reduce_min(self.X, axis=[1,2], keep_dims=True)
		self.max_ = tf.reduce_max(self.X, axis=[1,2], keep_dims=True)
		self.c1 = 2./(self.X - self.min_)
		self.c2 = 1.0 - self.c1 * self.max_
		self.X = self.c1*self.X + self.c2

	@scope
	def normalization_mean_std(self):
		mean, var = tf.nn.moments(self.X, axes=[1,2], keep_dims=True)
		self.X = (self.X - mean) / tf.sqrt(var)

	@scope
	def prediction(self):
		pass

	@scope
	def dilated(self):

		X_in = tf.expand_dims(self.X, 3)
		size_t = (self.args['chunk_size'] - (self.args['window_size'] - self.args['hop_size'])) / self.args['hop_size']
		size_f = self.args['window_size']//2 + 1
		
		X_in = tf.reshape(X_in, [-1, size_t, size_f, 1])

		f = 128
		y = tf.contrib.layers.conv2d(X_in, f, [1, 7], rate=[1,1])
		y = tf.contrib.layers.conv2d(y, f, [7, 1], rate=[1,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[4,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[8,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[16,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[32,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[1,1])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[2,2])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[4,4])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[8,8])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[16,16])
		y = tf.contrib.layers.conv2d(y, f, [5, 5], rate=[32,32])
		y = tf.contrib.layers.conv2d(y, 4, [5, 5], rate=[1,1])

		self.X = tf.reshape(y, [self.B, size_t, size_f*4])


	@scope
	def separate(self):
		# Input for KMeans algorithm [B, TF, E]
		input_kmeans = tf.reshape(self.prediction, [self.B, -1, self.embedding_size])
		self.embeddings = input_kmeans
		# S speakers to separate, give self.X in input not to consider silent bins
		kmeans = KMeans(nb_clusters=self.S, nb_tries=self.nb_tries, nb_iterations=self.nb_steps, 
			input_tensor=input_kmeans, beta=self.beta, latent_space_tensor=tf.abs(self.X_input) if self.with_silence else None,
			threshold=self.threshold)

		# Extract labels of each bins TF_i - labels [B, TF, 1]
		_ , labels = kmeans.network

		if self.beta is None:
			# It produces hard assignements [0,1,0, ...]
			self.masks = tf.one_hot(tf.cast(labels, tf.int32), self.S, 1.0, 0.0) # Create masks [B, TF, S]
		else:
			# It produces soft assignements
			self.masks = labels

		tf.summary.image('mask/predicted/1', tf.reshape(self.masks[:,:,0],[self.B, -1, self.F, 1]))
		tf.summary.image('mask/predicted/2', tf.reshape(self.masks[:,:,1],[self.B, -1, self.F, 1]))

		separated = tf.reshape(self.X_input, [self.B, -1, 1]) * self.masks # [B ,TF, S]
		separated = tf.reshape(separated, [self.B, -1, self.F, self.S])
		separated = tf.transpose(separated, [0,3,1,2]) # [B, S, T, F]
		separated = tf.reshape(separated, [self.B*self.S, -1, self.F, 1]) # [BS, T, F, 1]
		self.separated = separated
		return separated

	@scope
	def postprocessing(self):

		stft = tf.reshape(self.separated, [self.B*self.S, -1, self.F])
		
		stft = tf.complex(stft, 0.0*stft) * tf.exp(tf.complex(0.0, tf.angle(self.stfts)))

		istft = tf.contrib.signal.inverse_stft(
			stft, 
			frame_length=self.window_size, 
			frame_step=self.hop_size,
			window_fn=tf.contrib.signal.inverse_stft_window_fn(self.window_size-self.hop_size))
		output = tf.reshape(istft, [self.B, self.S, -1])
		self.output = output
		tf.summary.audio(name= "audio/output/reconstructed", tensor = tf.reshape(output, [-1, self.L]), sample_rate = config.fs, max_outputs=2)

		sdr_improvement, _ = self.sdr_improvement(self.x_non_mix, self.output)
		self.sdr_imp = sdr_improvement
		tf.summary.scalar('SDR_improvement', sdr_improvement)			

		return output

	@scope
	def cost_finetuning(self):
		perms = list(permutations(range(self.S))) # ex with 3: [0, 1, 2], [0, 2 ,1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]
		length_perm = len(perms)
		perms = tf.reshape(tf.constant(perms), [1, length_perm, self.S, 1])
		perms = tf.tile(perms, [self.B, 1, 1, 1])

		batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1, 1]), [1, length_perm, self.S, 1])
		perm_range = tf.tile(tf.reshape(tf.range(length_perm, dtype=tf.int32), shape=[1, length_perm, 1, 1]), [self.B, 1, self.S, 1])
		indicies = tf.concat([batch_range, perm_range, perms], axis=3)

		# [B, P, S, L]
		permuted_back = tf.gather_nd(tf.tile(tf.reshape(self.postprocessing, [self.B, 1, self.S, self.L]), [1, length_perm, 1, 1]), indicies) # 

		X_nmr = tf.reshape(self.x_non_mix, [self.B, 1, self.S, self.L])

		l2 = tf.reduce_sum(tf.square(X_nmr - permuted_back), axis=-1) # L2^2 norm
		l2 = tf.reduce_min(l2, axis=1) # Get the minimum over all possible permutations : B S
		l2 = tf.reduce_sum(l2, -1)
		l2 = tf.reduce_mean(l2, -1)
		tf.summary.scalar('cost', l2)

		return l2
 
	@scope
	def enhance(self):
		# [B, S, T, F]
		separated = tf.reshape(self.separate, [self.B, self.S, -1, self.F])
		
		# X [B, T, F]
		# Tiling the input S time - like [ a, b, c] -> [ a, a, b, b, c, c], not [a, b, c, a, b, c]
		X_in = tf.expand_dims(self.X_input, 1)
		X_in = tf.tile(X_in, [1, self.S, 1, 1])
		X_in = tf.reshape(X_in, [self.B, self.S, -1, self.F])

		# Concat the separated input and the actual tiled input
		sep_and_in = tf.concat([separated, X_in], axis = 3)
		sep_and_in = tf.reshape(sep_and_in, [self.B*self.S, -1, 2*self.F])
		
		if self.args['normalize_enhance']:
			mean, var = tf.nn.moments(sep_and_in, axes=[1,2], keep_dims=True)
			sep_and_in = (sep_and_in - mean) / tf.sqrt(var)

		layers = [
			BLSTM(self.args['layer_size_enhance'], drop_val=self.args["recurrent_dropout_enhance"],
				 name='BLSTM_'+str(i)) for i in range(self.args['nb_layers_enhance'])
		]

		layers += [
			Conv1D([1, self.args['layer_size_enhance'], self.F])
		]

		y = f_props(layers, sep_and_in)

		y = tf.reshape(y, [self.B, self.S, -1]) # [B, S, TF]

		tf.summary.image('mask/predicted/enhanced', tf.reshape(y, [self.B*self.S, -1, self.F, 1]))
		y = tf.transpose(y, [0, 2, 1]) # [B, TF, S]

		if self.args['nonlinearity'] == 'softmax':
			y = tf.nn.softmax(y)
		elif self.args['nonlinearity'] == 'tanh':
			y = tf.nn.tanh(y)

		self.enhanced_masks = tf.identity(y, name='enhanced_masks')
		
		tf.summary.image('mask/predicted/enhanced_soft', tf.reshape(tf.transpose(y, [0,2,1]), [self.B*self.S, -1, self.F, 1]))
		y = y * tf.reshape(self.X_input, [self.B, -1, 1]) # Apply enhanced filters # [B, TF, S] -> [BS, T, F, 1]

		self.cost_in = y

		y =  tf.transpose(y, [0, 2, 1])
		self.separated = y

		return tf.reshape(y , [self.B*self.S, -1, self.F, 1])

	@scope
	def enhance_cost(self):
		# Compute all permutations among the enhanced filters [B, TF, S] -> [B, TF, P, S]
		perms = list(permutations(range(self.S))) # ex with 3: [0, 1, 2], [0, 2 ,1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]
		length_perm = len(perms)

		# enhance [ B, TF, S] , X [B, T, F] -> [ B, TF, S]
		test_enhance = tf.tile(tf.reshape(tf.transpose(self.cost_in, [0,2,1]), [self.B, 1, self.S, -1]), [1, length_perm, 1, 1]) # [B, S, TF]
		
		perms = tf.reshape(tf.constant(perms), [1, length_perm, self.S, 1])
		perms = tf.tile(perms, [self.B, 1, 1, 1])

		batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1, 1]), [1, length_perm, self.S, 1])
		perm_range = tf.tile(tf.reshape(tf.range(length_perm, dtype=tf.int32), shape=[1, length_perm, 1, 1]), [self.B, 1, self.S, 1])
		indicies = tf.concat([batch_range, perm_range, perms], axis=3)

		# [B, P, S, TF]
		permuted_approx= tf.gather_nd(test_enhance, indicies)

		# X_non_mix [B, T, F, S]
		X_non_mix = tf.transpose(tf.reshape(self.X_non_mix, [self.B, 1, -1, self.S]), [0, 1, 3, 2])
		cost = tf.reduce_sum(tf.square(X_non_mix-permuted_approx), axis=-1) # Square difference on each bin 
		cost = tf.reduce_sum(cost, axis=-1) # Sum among all speakers

		cost = tf.reduce_min(cost, axis=-1) # Take the minimum permutation error

		cost = tf.reduce_mean(cost) #+ self.adapt_front.l * reg

		# tf.summary.scalar('regularization',  reg)
		tf.summary.scalar('cost', cost)

		return cost