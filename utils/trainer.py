# coding: utf-8
from data.dataset import H5PY_RW
from data.data_tools import read_metadata, males_keys, females_keys
from data.dataset import Mixer
import time
import numpy as np
import argparse
from utils.tools import getETA

class MyArgs(object):

	def __init__(self):
		
		parser = argparse.ArgumentParser(description="Argument Parser")

		# DataSet arguments
		parser.add_argument(
			'--dataset', help='Path to H5 dataset from workspace', required=False, default='h5py_files/train-clean-100-8-s.h5')
		parser.add_argument(
			'--chunk_size', type=int, help='Chunk size for inputs', required=False, default=20480)
		parser.add_argument(
			'--nb_speakers', type=int, help='Number of mixed speakers', required=False, default=2)
		parser.add_argument(
			'--no_random_picking', help='Do not pick random genders when mixing', action="store_false")
		parser.add_argument(
			'--validation_step',type=int, help='Nb of steps between each validation', required=False, default=1000)

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
			'--nb_layers', type=int, help='Number of stacked BLSTMs', required=False, default=3)
		self.parser.add_argument(
			'--layer_size', type=int, help='Size of hidden layers in BLSTM', required=False, default=600)
		self.parser.add_argument(
			'--embedding_size', type=int, help='Size of the embedding output', required=False, default=40)
		self.parser.add_argument(
			'--nonlinearity', help='Nonlinearity used', required=False, default='logistic')
		self.parser.add_argument(
			'--no_normalize', help='Normalization of the embedded space', action="store_false")

	def add_enhance_layer_args(self):
		self.parser.add_argument(
			'--nb_layers_enhance', type=int, help='Number of stacked BLSTMs for the enhance layer', required=False, default=3)
		self.parser.add_argument(
			'--layer_size_enhance', type=int, help='Size of hidden layers in BLSTM', required=False, default=600)

	def get_args(self):
		return self.parser.parse_args()

class Trainer(object):
	def __init__(self, trainer_type, **kwargs):

		H5_dic = read_metadata()
		males = H5PY_RW(kwargs['dataset'], subset = males_keys(H5_dic))
		fem = H5PY_RW(kwargs['dataset'], subset = females_keys(H5_dic))

		print 'Data with', len(H5_dic), 'male and female speakers'
		print males.length(), 'elements'
		print fem.length(), 'elements'

		self.mixed_data = Mixer([males, fem], chunk_size= kwargs['chunk_size'], 
			with_mask=False, with_inputs=True, shuffling=True,
			nb_speakers=kwargs['nb_speakers'], random_picking=kwargs['no_random_picking'])

		additional_args = {
			"tot_speakers" : len(H5_dic),
			"type" : trainer_type
		}

		kwargs.update(additional_args)
		self.args = kwargs

	def build_model(self):
		pass


	def train(self):

		print 'Total name :' 
		print self.model.runID

		batch_size_train = self.args['batch_size']
		batch_size_valid_test = batch_size_train

		# Get the number of batches in an epoch for each set (train/Valid/test)
		nb_batches_train = self.mixed_data.nb_batches(batch_size_train)
		self.mixed_data.select_split(1) # Switch on Validation set
		nb_batches_valid = self.mixed_data.nb_batches(batch_size_valid_test)
		self.mixed_data.select_split(2) # Switch on Test set
		nb_batches_test = self.mixed_data.nb_batches(batch_size_valid_test)
		self.mixed_data.select_split(0) # Switch back on Training set


		print '####TRAINING####'
		print nb_batches_train
		print '####VALID####'
		print nb_batches_valid
		print '####TEST####'
		print nb_batches_test

		nb_epochs = self.args['epochs']

		time_spent = [0 for _ in range(5)]

		best_validation_cost = 1e100

		for epoch in range(nb_epochs):
			for b in range(nb_batches_train):
				step = nb_batches_train*epoch + b
				X_non_mix, X_mix, I = self.mixed_data.get_batch(batch_size_train)

				t = time.time()
				c = self.model.train(X_mix, X_non_mix, I, step)
				t_f = time.time()
				time_spent = time_spent[1:] +[t_f-t]

				print 'Step #'  , step,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
					, nb_batches_train, b, nb_epochs, epoch)

				if step%self.args['validation_step'] == 0:
					t = time.time()
					# Select Validation set
					self.mixed_data.select_split(1)

					# Compute validation mean cost with batches
					costs = []
					for _ in range(nb_batches_valid):
						X_v_non_mix, X_v_mix, I = self.mixed_data.get_batch(batch_size_valid_test)

						cost = self.model.valid_batch(X_v_mix, X_v_non_mix, I)
						costs.append(cost)

					valid_cost = np.mean(costs)
					self.model.add_valid_summary(valid_cost, step)

					#Save model if it is better:
					if valid_cost < best_validation_cost:
						best_validation_cost = valid_cost # Save as new lowest cost
						best_path = self.model.save(step)
						print 'Save best model with :', best_validation_cost

					self.mixed_data.reset()
					self.mixed_data.select_split(0)

					t_f = time.time()
					print 'Validation set tested in ', t_f - t, ' seconds'
					print 'Validation set: ', valid_cost
			self.mixed_data.reset() # Reset the Training set from the beginning

		print 'Best model with Validation:  ', best_validation_cost
		print 'Path = ', best_path

		# Load the best model on validation set and test it
		self.model.restore_last_checkpoint()
		self.mixed_data.select_split(2)
		for _ in range(nb_batches_test):
			X_t_non_mix, X_t_mix, I = self.mixed_data.get_batch(batch_size_valid_test)

			cost = self.model.valid_batch(X_t_mix, X_t_non_mix)
			costs.append(cost)
		print 'Test cost = ', np.mean(costs)
		self.mixed_data.reset()

from models.adapt import Adapt

class STFT_Separator_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		self.separator = separator
		super(STFT_Separator_Trainer, self).__init__(trainer_type=name, **kwargs)

	def build_model(self):
		self.model = self.separator(**self.args)
		self.model.tensorboard_init()
		self.model.init_all()

class STFT_Separator_enhance_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		self.separator = separator
		super(STFT_Separator_enhance_Trainer, self).__init__(trainer_type=name, **kwargs)

	def build_model(self):
		self.model = self.separator.load(self.args['model_folder'], self.args)
		self.model.restore_model(self.args['model_folder'])
		self.model.add_enhance_layer()
		self.model.tensorboard_init()
		# Initialize only non restored values
		self.model.initialize_non_init()


class Adapt_Pretrainer(Trainer):
	def __init__(self, **kwargs):
		super(Adapt_Pretrainer, self).__init__(trainer_type='pretraining', **kwargs)

	def build_model(self):
		self.model = Adapt(**self.args)
		self.model.tensorboard_init()
		self.model.init_all()

class Front_Separator_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		super(Front_Separator_Trainer, self).__init__(trainer_type=name, **kwargs)
		self.separator = separator

	def build_model(self):
		self.model = Adapt.load(self.args['model_folder'], self.args)
		self.model.restore_model(self.args['model_folder'])
		self.model.connect_only_front_to_separator(self.separator)
		# Initialize only non restored values
		self.model.initialize_non_init()

class Front_Separator_Finetuning_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		super(Front_Separator_Finetuning_Trainer, self).__init__(trainer_type=name, **kwargs)
		self.separator = separator

	def build_model(self):
		self.model = Adapt.load(self.args['model_folder'], self.args)
		
		# Expanding the graph with enhance layer
		with self.model.graph.as_default() : 
			self.model.connect_front(self.separator)
			self.model.sepNet.output = self.model.sepNet.separate
			self.model.back
			self.model.restore_model(self.args['model_folder'])
			self.model.cost_model = self.model.cost
			self.model.finish_construction()
			self.model.optimize
		self.model.tensorboard_init()
		# Initialize only non restored values
		self.model.initialize_non_init()

class Front_Separator_Enhance_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		super(Front_Separator_Enhance_Trainer, self).__init__(trainer_type=name, **kwargs)
		self.separator = separator

	def build_model(self):
		self.model = Adapt.load(self.args['model_folder'], self.args)
		# Restoring previous Model:
		self.model.restore_front_separator(self.args['model_folder'], self.separator)
		# Expanding the graph with enhance layer
		with self.model.graph.as_default() : 
			self.model.sepNet.output = self.model.sepNet.enhance
			self.model.cost_model = self.model.sepNet.enhance_cost
			self.model.finish_construction()
			self.model.freeze_all_except('enhance')
			self.model.optimize
		self.model.tensorboard_init()
		# Initialize only non restored values
		self.model.initialize_non_init()

class Front_Separator_Enhance_Finetuning_Trainer(Trainer):
	def __init__(self, separator, name, **kwargs):
		super(Front_Separator_Enhance_Finetuning_Trainer, self).__init__(trainer_type=name, **kwargs)
		self.separator = separator

	def build_model(self):
		self.model = Adapt.load(self.args['model_folder'], self.args)
		# Restoring the front layer:

		# Expanding the graph with enhance layer
		with self.model.graph.as_default() : 
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