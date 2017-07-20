from data.dataset import H5PY_RW, Mixer
from models.das import DAS
from data.data_tools import read_data_header, males_keys, females_keys
from utils.audio import istft_

import config
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

	H5_dico = read_data_header()

	Males = H5PY_RW()
	Males.open_h5_dataset('test.h5py', subset = males_keys(H5_dico))
	Males.set_chunk(config.chunk_size)
	Males.shuffle()
	print 'Male voices loaded: ', Males.length(), ' items'

	Females = H5PY_RW()
	Females.open_h5_dataset('test.h5py', subset = females_keys(H5_dico))
	Females.set_chunk(config.chunk_size)
	Females.shuffle()
	print 'Female voices loaded: ', Females.length(), ' items'

	Mixer = Mixer([Males, Females])
	# Mixer.select_split(2)
	das_model = DAS(S=len(Mixer.get_labels()), T= config.chunk_size)
	print 'Model DAS created'
	das_model.init()

	Mixer.select_split(1)
	#Validation Data
	X_valid, Y_valid, Ind_valid = Mixer.get_batch(1)
	X_raw_valid =[]
	for x in X_valid:
			_, x_recons = istft_(x.T)
			X_raw_valid.append(x_recons)
	X_valid = X_valid[:,:,:128]
	Y_valid = Y_valid[:,:,:128,:]
	X_valid = np.sqrt(np.abs(X_valid))
	X_valid = (X_valid - X_valid.min())/(X_valid.max() - X_valid.min())

	cost_valid_min = 1e10
	Mixer.select_split(0)

	for i in range(config.max_iterations):
		print 'Step #' ,i
		X, Y, Ind = Mixer.get_batch(1)
		x_mixture =[]

		for x in X:
			_, x_recons = istft_(x.T)
			x_mixture.append(x_recons)

		X = X[:,:,:128]
		Y = Y[:,:,:128,:]

		# Scale the model inputs
		X = np.sqrt(np.abs(X))
		X = (X - X.min())/(X.max() - X.min())

		das_model.train(X, Y, Ind, x_mixture, i)

		if (i+1) % config.batch_test == 0:

			# Cost obtained with the current model on the validation set
			cost_valid = das_model.valid(X_valid, X_raw_valid, Y_valid, Ind_valid, i)
			
			if cost_valid < cost_valid_min:
				print 'DAS model saved at iteration number ', i,' with cost = ', cost_valid 
				cost_valid_min = cost_valid
				das_model.save(i)
				last_saved = i

			if i - last_saved > config.stop_iterations:
				print 'Stop'
				break

			