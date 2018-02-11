# coding: utf-8

from data.dataset import H5PY_RW
from data.data_tools import read_metadata, males_keys, females_keys
from data.dataset import Mixer
from models.L41 import L41Model
from utils.tools import getETA
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
		"tot_speakers" : len(H5_dic),
		"freqs_size" : args.window_size//2 +1,
		"type" : "STFT_L41_enhance"
	}

	d = vars(args)
	d.update(additional_args)

	####
	model = L41Model.load(args.model_folder, d)
	model.restore_model(args.model_folder)
	model.add_enhance_layer()
	model.tensorboard_init()

	# Initialize only non restored values
	model.initialize_non_init()

	print 'Total name :' 
	print model.runID

	batch_size_train = args.batch_size
	batch_size_valid_test = batch_size_train

	#
	# Get the number of batches in an epoch for each set (train/Valid/test)
	#
	nb_batches_train = mixed_data.nb_batches(batch_size_train)
	mixed_data.select_split(1) # Switch on Validation set
	nb_batches_valid = mixed_data.nb_batches(batch_size_valid_test)
	mixed_data.select_split(2) # Switch on Test set
	nb_batches_test = mixed_data.nb_batches(batch_size_valid_test)
	mixed_data.select_split(0) # Switch back on Training set


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

			t = time.time()
			c = model.train(X_mix, X_non_mix, I, step)
			t_f = time.time()
			time_spent = time_spent[1:] +[t_f-t]

			print 'Step #'  , step,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
				, nb_batches_train, b, nb_epochs, epoch)

			if step%args.validation_step == 0:
				t = time.time()
				# Select Validation set
				mixed_data.select_split(1)

				# Compute validation mean cost with batches
				costs = []
				for _ in range(nb_batches_valid):
					X_v_non_mix, X_v_mix, I = mixed_data.get_batch(batch_size_valid_test)

					cost = model.valid_batch(X_v_mix, X_v_non_mix, I)
					costs.append(cost)

				valid_cost = np.mean(costs)
				model.add_valid_summary(valid_cost, step)

				#Save model if it is better:
				if valid_cost < best_validation_cost:
					best_validation_cost = valid_cost # Save as new lowest cost
					best_path = model.save(step)
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
	model.restore_last_checkpoint()
	mixed_data.select_split(2)
	for _ in range(nb_batches_test):
		X_t_non_mix, X_t_mix, I = mixed_data.get_batch(batch_size_valid_test)

		cost = model.valid_batch(X_t_mix, X_t_non_mix, I)
		costs.append(cost)
	print 'Test cost = ', np.mean(costs)
	mixed_data.reset()


###
### EXAMPLE
# python -m experiments.pretraining --dataset h5py_files/train-clean-100-8-s.h5 --chunk_size 5120 --nb_speakers 2 --epochs 1 --batch_size 3 --learning_rate 0.01 --window_size 1024 --max_pool 256 --filters 512 --regularization 1e-4 --beta 0.01 --sparsity 0.1 --no_random_picking
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
	
	# L41 model to load + params
	parser.add_argument(
		'--model_folder', help='Path to the Model folder to load', required=True)

	# Training arguments
	parser.add_argument(
		'--epochs', type=int, help='Number of epochs', required=False, default=10)
	parser.add_argument(
		'--batch_size', type=int, help='Batch size', required=False, default=64)
	parser.add_argument(
		'--learning_rate', type=float, help='learning rate for training', required=False, default=0.1)
	
	#Preprocess arguments
	parser.add_argument(
		'--window_size', type=int, help='Size of the window for STFT', required=False, default=512)
	parser.add_argument(
		'--hop_size', type=int, help='Hop size for the STFT', required=False, default=256)

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
