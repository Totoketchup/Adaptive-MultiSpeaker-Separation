from data.dataset import H5PY_RW, Mixer
from models.adapt import Adapt
from models.dpcl import DPCL
from data.data_tools import read_data_header, males_keys, females_keys
import config
import numpy as np
# import tensorflow as tf
# import soundfile as sf

def normalize(y):
	y = y - np.mean(y)
	return y/np.std(y)

if __name__ == "__main__":

	H5_dico = read_data_header()

	males = H5PY_RW()
	males.open_h5_dataset('test_raw.h5py', subset = males_keys(H5_dico))
	males.set_chunk(5*4*512)
	males.shuffle()
	print 'Male voices loaded: ', males.length(), ' items'

	fem = H5PY_RW()
	fem.open_h5_dataset('test_raw.h5py', subset = females_keys(H5_dico))
	fem.set_chunk(5*4*512)
	fem.shuffle()
	print 'Female voices loaded: ', fem.length(), ' items'

	Mixer = Mixer([males, fem], with_mask=False, with_inputs=True)

	adapt_model = Adapt.load('jolly-firefly-9628', pretraining=False, separator=DPCL)
	# adapt_model.init()
	print 'Model DAS created'

	testVar = raw_input("Model loaded : Press Enter")


	cost_valid_min = 1e10
	Mixer.select_split(0)
	learning_rate = 0.01

	for i in range(config.max_iterations):
		X_in, X_mix, Ind = Mixer.get_batch(1)
		if (i+1)%100 == 0:
			learning_rate /= 10
		c = adapt_model.train(X_mix, X_in,learning_rate, i)
		print 'Step #'  ,i,' ', c 

		if i%20 == 0: #cost_valid < cost_valid_min:
			print 'DAS model saved at iteration number ', i,' with cost = ', c 
			adapt_model.save(i)
