# coding: utf-8
from data.dataset import H5PY_RW
from data.data_tools import read_metadata, males_keys, females_keys
from data.dataset import Mixer
from models.adapt import Adapt
from models.dpcl import DPCL
from utils.tools import getETA, normalize_mix
import time
import numpy as np
import argparse

def main(args):

	H5_dic = read_metadata()

	males = H5PY_RW(args.dataset, subset = males_keys(H5_dic))
	fem = H5PY_RW(args.dataset, subset = females_keys(H5_dic))

	print 'Data with', len(H5_dic), 'male and female speakers'
	print males.length(), 'elements'
	print fem.length(), 'elements'

	mixed_data = Mixer([males, fem], chunk_size=args.chunk_size, 
		with_mask=False, with_inputs=True, shuffling=True,
		nb_speakers=args.nb_speakers, random_picking=args.no_random_picking)

	additional_args = {
		"type" : "front_DPCL_enhance",
		"pretraining": False,
		"separator": None,
		"tot_speakers" : len(H5_dic),
	}

	d = vars(args)
	d.update(additional_args)

	####
	adapt_model = Adapt.load(args.model_folder, d)

	# Restoring previous Model:
	adapt_model.restore_front_separator(args.model_folder, DPCL)
	
	# Expanding the graph with enhance layer
	with adapt_model.graph.as_default() : 
		adapt_model.sepNet.output = adapt_model.sepNet.enhance
		adapt_model.cost = adapt_model.sepNet.enhance_cost
		adapt_model.freeze_all_except('enhance')
		adapt_model.optimize

	adapt_model.tensorboard_init()

	# Initialize only non restored values
	adapt_model.initialize_non_init()

	print 'Total name :' 
	print adapt_model.runID

	batch_size_train = args.batch_size
	batch_size_valid_test = batch_size_train

	# Get the number of batches in an epoch for each set (train/Valid/test)
	nb_batches_train = mixed_data.nb_batches(batch_size_train)
	mixed_data.select_split(1) # Switch on Validation set
	nb_batches_valid = mixed_data.nb_batches(batch_size_valid_test)
	mixed_data.select_split(2) # Switch on Test set
	nb_batches_test = mixed_data.nb_batches(batch_size_valid_test)
	mixed_data.select_split(0) # Switch back on Training set

	print '#BATCH IN EPOCH#'
	print '####TRAINING####'
	print nb_batches_train
	print '####VALID####'
	print nb_batches_valid
	print '####TEST####'
	print nb_batches_test

	nb_epochs = args.epochs

	time_spent = [0 for _ in range(5)]

	best_validation_cost = 1e100

	for epoch in range(nb_epochs):
		for b in range(nb_batches_train):
			step = nb_batches_train*epoch + b
			X_non_mix, X_mix, I = mixed_data.get_batch(batch_size_train)
			X_mix, X_non_mix = normalize_mix(X_mix, X_non_mix)

			t = time.time()
			c = adapt_model.train(X_mix, X_non_mix, args.learning_rate, step, I=I)
			t_f = time.time()
			time_spent = time_spent[1:] +[t_f-t]

			print 'Step #'  , step,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
				, nb_batches_train, b, nb_epochs, epoch)

			if step%1000 == 0:
				t = time.time()
				# Select Validation set
				mixed_data.select_split(1)

				# Compute validation mean cost with batches
				costs = []
				for _ in range(nb_batches_valid):
					X_v_non_mix, X_v_mix, I = mixed_data.get_batch(batch_size_valid_test)
					X_v_mix, X_v_non_mix = normalize_mix(X_v_mix, X_v_non_mix)

					cost = adapt_model.valid_batch(X_v_mix, X_v_non_mix, I=I)
					costs.append(cost)

				valid_cost = np.mean(costs)
				adapt_model.add_valid_summary(valid_cost, step)

				#Save model if it is better:
				if valid_cost < best_validation_cost:
					best_validation_cost = valid_cost # Save as new lowest cost
					best_path = adapt_model.save(step)
					print 'Save best model with :', best_validation_cost

				mixed_data.reset()
				mixed_data.select_split(0)

				t_f = time.time()
				print 'Validation set tested in ', t_f - t, ' seconds'
				print 'Validation set: ', valid_cost
		mixed_data.reset() # Reset the Training set from the beginning

	print 'Best model with Validation:  ', best_validation_cost
	print 'Path = ', best_path

	# Load the best model on validation set and test it
	adapt_model.restore_last_checkpoint()
	mixed_data.select_split(2)
	for _ in range(nb_batches_test):
		X_t_non_mix, X_t_mix, I = mixed_data.get_batch(batch_size_valid_test)
		X_t_mix, X_t_non_mix = normalize_mix(X_t_mix, X_t_non_mix, I=I)

		cost = adapt_model.valid_batch(X_t_mix, X_t_non_mix)
		costs.append(cost)
	print 'Test cost = ', np.mean(costs)
	mixed_data.reset()
###
### EXAMPLE
# python -m experiments.L41_front --adapt_folder log/pretraining/AdaptiveNet-sparkling-sky-7106 --batch_size 2 --learning_rate 0.1
###
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Adaptive Layer Pretraining')

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

	# Adapt model to load + params
	parser.add_argument(
		'--model_folder', help='Path to the Model folder to load', required=True)

	# Training arguments
	parser.add_argument(
		'--epochs', type=int, help='Number of epochs', required=False, default=10)
	parser.add_argument(
		'--batch_size', type=int, help='Batch size', required=False, default=64)
	parser.add_argument(
		'--learning_rate', type=float, help='learning rate for training', required=False, default=0.1)
	
	#Network arguments
	parser.add_argument(
		'--layer_size', type=int, help='Size of hidden layers in BLSTM', required=False, default=600)
	parser.add_argument(
		'--embedding_size', type=int, help='Size of the embedding output', required=False, default=40)
	parser.add_argument(
		'--nonlinearity', help='Nonlinearity used', required=False, default='logistic')
	parser.add_argument(
		'--normalize', help='Normalization of the embedded space', action="store_false")


	args = parser.parse_args()
	print args
	main(args)
