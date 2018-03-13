# coding: utf-8
from data.dataset import Dataset
import time
import numpy as np
import argparse
from utils.tools import getETA
from utils.ops import normalize_mix

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
			'--nb_layers', type=int, help='Number of stacked BLSTMs', required=False, default=3)
		self.parser.add_argument(
			'--layer_size', type=int, help='Size of hidden layers in BLSTM', required=False, default=600)
		self.parser.add_argument(
			'--embedding_size', type=int, help='Size of the embedding output', required=False, default=40)
		self.parser.add_argument(
			'--no_normalize', help='Normalization of the embedded space', action="store_false")

	def add_enhance_layer_args(self):
		self.parser.add_argument(
			'--nb_layers_enhance', type=int, help='Number of stacked BLSTMs for the enhance layer', required=False, default=3)
		self.parser.add_argument(
			'--layer_size_enhance', type=int, help='Size of hidden layers in BLSTM', required=False, default=600)
		self.parser.add_argument(
			'--nonlinearity', help='Nonlinearity used in output', choices=['tanh', 'softmax'], required=False, default='softmax')

	def add_adapt_args(self):
		#Preprocess arguments
		self.parser.add_argument(
			'--window_size', type=int, help='Size of the 1D Conv width', required=False, default=1024)
		self.parser.add_argument(
			'--filters', type=int, help='Number of filters/bases for the 1D Conv', required=False, default=512)
		self.parser.add_argument(
			'--max_pool', type=int, help='Max Pooling size', required=False, default=512)

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

		self.dataset = Dataset(**kwargs)

		additional_args = {
			"tot_speakers" : self.dataset.tot_speakers,
			"type" : trainer_type
		}

		kwargs.update(additional_args)
		self.args = kwargs

	def build_model(self):
		pass

	def train(self):

		print 'Total name :' 
		print self.model.runID

		nb_epochs = self.args['epochs']
		batch_size = self.args['batch_size']
		time_spent = [0 for _ in range(5)]
		nb_batches = self.dataset.nb_batch(batch_size)

		best_validation_cost = 1e100

		step = 0
		for epoch in range(nb_epochs):
			for b ,(x_mix, x_non_mix, I) in enumerate(self.dataset.get_batch(self.dataset.TRAIN, batch_size)):

				# x_mix, x_non_mix, _, _ = normalize_mix(x_mix, x_non_mix, type_='mean-std')

				t = time.time()
				c = self.model.train(x_mix, x_non_mix, I, step)
				t_f = time.time()
				time_spent = time_spent[1:] +[t_f-t]

				print 'Epoch #', epoch+1,'Step #', step+1,' loss=', c \
					, ' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent)) \
					, nb_batches, b, nb_epochs, epoch)

				if step%self.args['validation_step'] == 0:
					t = time.time()
					# Compute validation mean cost with batches to avoid memory problems
					costs = []
					for x_mix_v, x_non_mix_v, I_v in self.dataset.get_batch(self.dataset.VALID, batch_size):

						# x_mix_v, x_non_mix_v, _, _ = normalize_mix(x_mix_v, x_non_mix_v, type_='mean-std')

						cost = self.model.valid_batch(x_mix_v, x_non_mix_v, I_v, step)
						costs.append(cost)

					valid_cost = np.mean(costs)
					self.model.add_valid_summary(valid_cost, step)

					# Save the model if it is better:
					if valid_cost < best_validation_cost:
						best_validation_cost = valid_cost # Save as new lowest cost
						best_path = self.model.save(step)
						print 'Save best model with :', best_validation_cost

					t_f = time.time()
					print 'Validation set tested in ', t_f - t, ' seconds'
					print 'Validation set: ', valid_cost
				
				step += 1

		print 'Best model with Validation:  ', best_validation_cost
		print 'Path = ', best_path

		# Load the best model on validation set and test it
		self.model.restore_last_checkpoint()
		
		for x_mix_t, x_non_mix_t, I_t in self.dataset.get_batch(self.dataset.TEST, batch_size):
			# x_mix_t, x_non_mix_t, _, _ = normalize_mix(x_mix_t, x_non_mix_t)
			cost = self.model.test_batch(x_mix_t, x_non_mix_t, I_t)
			costs.append(cost)
		print 'Test cost = ', np.mean(costs)

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