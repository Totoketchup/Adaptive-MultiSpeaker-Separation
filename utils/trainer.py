# coding: utf-8
from data.dataset import Dataset, TFDataset, MixGenerator
import time
import numpy as np
import argparse
from utils.tools import getETA
from utils.ops import normalize_mix
import tensorflow as tf
from models.adapt import Adapt

class MyArgs(object):

	def __init__(self):
		
		parser = argparse.ArgumentParser(description="Argument Parser")

		# DataSet arguments
		parser.add_argument(
			'--dataset', help='Path to H5 dataset from workspace', required=False, default='h5py_files/train-clean-100-8-s.h5')
		parser.add_argument(
			'--dataset_normalize', help='Mean/Std normalization for each input', action="store_true")
		parser.add_argument(
			'--chunk_size', type=int, help='Chunk size for inputs', required=False, default=20480)
		parser.add_argument(
			'--nb_speakers', type=int, help='Number of mixed speakers', required=False, default=2)
		parser.add_argument(
			'--no_random_picking', help='Do not pick random genders when mixing', action="store_true")
		parser.add_argument(
			'--validation_step',type=int, help='Nb of steps between each validation', required=False, default=1000)
		parser.add_argument(
			'--men', help='Use men voices', action="store_true")
		parser.add_argument(
			'--women', help='Use women voices', action="store_true")
		
		# Training arguments
		parser.add_argument(
			'--epochs', type=int, help='Number of epochs', required=False, default=10)
		parser.add_argument(
			'--batch_size', type=int, help='Batch size', required=False, default=64)
		parser.add_argument(
			'--learning_rate', type=float, help='learning rate for training', required=False, default=0.1)

		self.parser = parser

	def add_stft_args(self):
		self.parser.add_argument(
			'--window_size', type=int, help='Size of the window for STFT', required=False, default=512)
		self.parser.add_argument(
			'--hop_size', type=int, help='Hop size for the STFT', required=False, default=256)

	def add_separator_args(self):
		self.parser.add_argument(
			'--normalize_separator', help='Normalize the input of the separator', choices=['None','01', 'meanstd'], required=False, default='None')
		self.parser.add_argument(
			'--nb_layers', type=int, help='Number of stacked BLSTMs', required=False, default=3)
		self.parser.add_argument(
			'--layer_size', type=int, help='Size of hidden layers in BLSTM', required=False, default=600)
		self.parser.add_argument(
			'--embedding_size', type=int, help='Size of the embedding output', required=False, default=40)
		self.parser.add_argument(
			'--nb_tries', type=int, help='Number of tries for KMEANS', required=False, default=10)
		self.parser.add_argument(
			'--nb_steps', type=int, help='Number of steps for KMEANS', required=False, default=10)
		self.parser.add_argument(
			'--no_normalize', help='Normalization of the embedded space', action="store_false")
		self.parser.add_argument(
			'--abs_input', help='tf.abs on the L41 input', action="store_true")
		self.parser.add_argument(
			'--beta_kmeans', type=float, help='Beta value for KMEANS - None = Hard KMEANS', required=False, default=None)
		self.parser.add_argument(
			'--threshold', type=float, help='Threshold for the silent bins', required=False, default=2.0)
		self.parser.add_argument(
			'--with_silence', help='Silence weak bins during KMEANS', action="store_true")
		self.parser.add_argument(
			'--silence_loss', help='Silence weak bins in the loss function', action="store_true")
		self.parser.add_argument(
			'--threshold_silence_loss', type=float, help='Threshold for the silent bins', required=False, default=2.0)
		self.parser.add_argument(
			'--function_mask', help='#TODO', choices=['None','linear', 'sqrt', 'square'], required=False, default='None')
		
	def add_enhance_layer_args(self):
		self.parser.add_argument(
			'--normalize_enhance', help='Normalize the input of the enhance layer', action="store_true")
		self.parser.add_argument(
			'--nb_layers_enhance', type=int, help='Number of stacked BLSTMs for the enhance layer', required=False, default=3)
		self.parser.add_argument(
			'--layer_size_enhance', type=int, help='Size of hidden layers in BLSTM', required=False, default=600)
		self.parser.add_argument(
			'--nonlinearity', help='Nonlinearity used in output', choices=['tanh', 'softmax', 'None'], required=False, default='softmax')

	def add_adapt_args(self):
		#Preprocess arguments
		self.parser.add_argument(
			'--window_size', type=int, help='Size of the 1D Conv width', required=False, default=1024)
		self.parser.add_argument(
			'--filters', type=int, help='Number of filters/bases for the 1D Conv', required=False, default=512)
		self.parser.add_argument(
			'--max_pool', type=int, help='Max Pooling size', required=False, default=512)
		self.parser.add_argument(
			'--with_max_pool', help='Use Max pooling and not hop', action="store_true")
		self.parser.add_argument(
			'--with_average_pool', help='Use Average pooling and not hop', action="store_true")

		#Loss arguments
		self.parser.add_argument(
			'--regularization', type=float, help='Coefficient for L2 regularization', required=False, default=1e-4)
		self.parser.add_argument(
			'--beta', type=float, help='Coefficient for Sparsity constraint', required=False, default=1e-2)
		self.parser.add_argument(
			'--sparsity', type=float, help='Average Sparsity constraint', required=False, default=0.01)
		self.parser.add_argument(
			'--overlap_coef', type=float, help='Coefficient for Overlapping loss', required=False, default=0.001)
		self.parser.add_argument(
			'--overlap_value', type=float, help='Coefficient for Overlapping loss', required=False, default=0.1)
		self.parser.add_argument(
			'--loss', choices=['l2', 'sdr', 'l2+sdr', 'sdr+l2'], required=False, default='sdr')
		self.parser.add_argument(
			'--separation', choices=['perfect', 'mask'], required=False, default='perfect')

	def get_args(self):
		parsed = self.parser.parse_args()

		sex = []
		if parsed.men : sex.append('M')
		if parsed.women : sex.append('F')
		parsed.sex = sex

		return parsed 

class Trainer(object):
	def __init__(self, trainer_type, **kwargs):

		# self.dataset = Dataset(**kwargs)

		self.batch_size = kwargs['batch_size']
		
		additional_args = {
			"type" : trainer_type
		}

		kwargs.update(additional_args)
		self.args = kwargs

	def train(self):

		print 'Total name :' 

		nb_epochs = self.args['epochs']
		time_spent = [0 for _ in range(10)]
		
		with tf.Graph().as_default() as graph:
			tfds = TFDataset(**self.args)

			additional_args = {
				"mix": tfds.next_mix,
				"non_mix": tfds.next_non_mix,
				"ind": tfds.next_ind,
				"pipeline": True,
				"tot_speakers" : 251
			}

			self.args.update(additional_args)

			config_ = tf.ConfigProto()
			config_.gpu_options.allow_growth = True
			config_.allow_soft_placement = True

			with tf.Session(graph=graph, config=config_).as_default() as sess:

				self.build()

				tfds.init_handle()

				nb_batches_train = tfds.length('train')
				nb_batches_test = tfds.length('test')
				nb_batches_valid = tfds.length('valid')

				print 'BATCHES'
				print nb_batches_train, nb_batches_test, nb_batches_valid

				feed_dict_train = {tfds.handle: tfds.get_handle('train')}
				feed_dict_valid = {tfds.handle: tfds.get_handle('valid')}
				feed_dict_test = {tfds.handle: tfds.get_handle('test')}

				best_validation_cost = 1e100

				t1 = time.clock()

				step = 0

				for epoch in range(nb_epochs):
					sess.run(tfds.training_initializer)

					for b in range(nb_batches_train):

						t = time.clock()

						c = self.model.train(feed_dict_train, step)
										
						if (step+1)%self.args['validation_step'] == 0:
							t = time.clock()

							sess.run(tfds.validation_initializer)

							# Compute validation mean cost with batches to avoid memory problems
							costs = []

							for b_v in range(nb_batches_valid):
								cost = self.model.valid_batch(feed_dict_valid,step)
								costs.append(cost)

							valid_cost = np.mean(costs)
							self.model.add_valid_summary(valid_cost, step)

							# Save the model if it is better:
							if valid_cost < best_validation_cost:
								best_validation_cost = valid_cost # Save as new lowest cost
								best_path = self.model.save(step)
								print 'Save best model with :', best_validation_cost

							t_f = time.clock()
							print 'Validation set tested in ', t_f - t, ' seconds'
							print 'Validation set: ', valid_cost

						time_spent = time_spent[1:] +[time.clock()-t1]
						avg =  sum(time_spent)/len(time_spent)
						print 'Epoch #', epoch+1,'/', nb_epochs,' Batch #', b+1,'/',nb_batches_train,'in', avg,'sec loss=', c \
							, ' ETA = ', getETA(avg, nb_batches_train, b+1, nb_epochs, epoch+1)

						t1 = time.clock()

						step += 1

				print 'Best model with Validation:  ', best_validation_cost
				print 'Path = ', best_path

				# Load the best model on validation set and test it
				self.model.restore_last_checkpoint()
				
				sess.run(tfds.test_initializer)

				for b_t in range(nb_batches_test):
					cost = self.model.test_batch(feed_dict_test)
					costs.append(cost)
				print 'Test cost = ', np.mean(costs)


class STFT_Separator_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		self.separator = separator
		super(STFT_Separator_Trainer, self).__init__(trainer_type=name, **kwargs)

	def build(self):
		self.model = self.separator(**self.args)
		self.model.tensorboard_init()
		self.model.init_all()		

class STFT_Separator_enhance_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		self.separator = separator
		super(STFT_Separator_enhance_Trainer, self).__init__(trainer_type=name, **kwargs)
		
	def build(self):
		self.model = self.separator.load(self.args['model_folder'], self.args)
		self.model.restore_model(self.args['model_folder'])
		self.model.add_enhance_layer()
		self.model.tensorboard_init()
		# Initialize only non restored values
		self.model.initialize_non_init()

class STFT_Separator_FineTune_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		self.separator = separator
		super(STFT_Separator_FineTune_Trainer, self).__init__(trainer_type=name, **kwargs)
		
	def build(self):
		self.model = self.separator.load(self.args['model_folder'], self.args)
		# self.model.init_separator()
		self.model.create_saver()
		self.model.restore_model(self.args['model_folder'])
		self.model.add_finetuning()
		self.model.tensorboard_init()
		# Initialize only non restored values
		self.model.initialize_non_init()		


class Adapt_Pretrainer(Trainer):

	def __init__(self, **kwargs):
		super(Adapt_Pretrainer, self).__init__(trainer_type='pretraining', **kwargs)

	def build(self):
		self.model = Adapt(**self.args)
		self.model.tensorboard_init()
		self.model.init_all()

class Front_Separator_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		super(Front_Separator_Trainer, self).__init__(trainer_type=name, **kwargs)
		self.separator = separator

	def build(self):
		self.model = Adapt.load(self.args['model_folder'], self.args)
		self.model.connect_only_front_to_separator(self.separator)
		# Initialize only non restored values
		self.model.initialize_non_init()

class Front_Separator_Finetuning_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		super(Front_Separator_Finetuning_Trainer, self).__init__(trainer_type=name, **kwargs)
		self.separator = separator

	def build(self):
		self.model = Adapt.load(self.args['model_folder'], self.args)
		# Expanding the graph with enhance layer
		self.model.connect_front(self.separator)
		self.model.sepNet.output = self.model.sepNet.separate
		self.model.back
		self.model.create_saver()
		self.model.restore_model(self.args['model_folder'])
		self.model.cost_model = self.model.cost
		self.model.finish_construction()
		self.model.freeze_all_except('prediction', 'speaker_centroids')
		self.model.optimize
		self.model.tensorboard_init()
		# Initialize only non restored values
		self.model.initialize_non_init()

class Front_Separator_Enhance_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		super(Front_Separator_Enhance_Trainer, self).__init__(trainer_type=name, **kwargs)
		self.separator = separator

	def build(self):
		self.model = Adapt.load(self.args['model_folder'], self.args)
		# Restoring previous Model:
		self.model.connect_enhance_to_separator(self.separator)
		self.model.initialize_non_init()

class Front_Separator_Enhance_Finetuning_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		super(Front_Separator_Enhance_Finetuning_Trainer, self).__init__(trainer_type=name, **kwargs)
		self.separator = separator

	def build(self):
		self.model = Adapt.load(self.args['model_folder'], self.args)
		# Restoring the front layer:
		# Expanding the graph with enhance layer
		self.model.connect_front(self.separator)
		self.model.sepNet.output = self.model.sepNet.enhance
		self.model.back
		self.model.restore_model(self.args['model_folder'])
		self.model.cost_model = self.model.cost
		self.model.finish_construction()
		self.model.optimize
		self.model.tensorboard_init()
		# Initialize only non restored values
		self.model.initialize_non_init()