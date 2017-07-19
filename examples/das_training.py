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

	das_model = DAS(S=len(Mixer.get_labels()), T= config.chunk_size)
	print 'Model DAS created'
	das_model.init()


	for i in range(100):
		print 'Step #' ,i
		X, Y, Ind = Mixer.get_batch(64)
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
		das_model.save(i)