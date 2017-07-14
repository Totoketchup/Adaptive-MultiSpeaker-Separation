from data.dataset import H5PY_RW, Mixer
from models.das import DAS
import config
import numpy as np
import tensorflow as tf
if __name__ == "__main__":
	Males = H5PY_RW()
	Males.open_h5_dataset('test.h5py')
	Males.set_chunk(config.chunk_size)

	Females = H5PY_RW()
	Females.open_h5_dataset('test.h5py')
	Females.set_chunk(config.chunk_size)
	Females.shuffle()

	Mixer = Mixer([Males, Females])
	X, Y, Ind = Mixer.get_batch(10)

	das_model = DAS(S=9026, T= config.chunk_size)

	# Merge all the summaries and write them out to /log (by default)

	das_model.init()

	for i in range(100):
		print 'Step #' ,i
		X, Y, Ind = Mixer.get_batch(32)
		X = X[:,:,:128]
		Y = Y[:,:,:128,:]+
		# Scale the model inputs
		X = np.sqrt(X)
		X = (X - X.min())/(X.max() - X.min())

		das_model.train(X, Y, Ind, i)