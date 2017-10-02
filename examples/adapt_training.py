from data.dataset import H5PY_RW, Mixer
from models.adapt import Adapt
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

	Males = H5PY_RW()
	Males.open_h5_dataset('test_raw.h5py', subset = males_keys(H5_dico))
	Males.set_chunk(5*9*512)
	Males.shuffle()
	print 'Male voices loaded: ', Males.length(), ' items'

	Females = H5PY_RW()
	Females.open_h5_dataset('test_raw.h5py', subset = females_keys(H5_dico))
	Females.set_chunk(5*9*512)
	Females.shuffle()
	print 'Female voices loaded: ', Females.length(), ' items'

	Mixer = Mixer([Males, Females], with_mask=False, with_inputs=True)
	# Mixer.select_split(2)
	adapt_model = Adapt()
	print 'Model DAS created'
	adapt_model.init()

	cost_valid_min = 1e10
	Mixer.select_split(0)
	learning_rate = 0.0005
	for i in range(config.max_iterations):
		X_in, X_mix, Ind = Mixer.get_batch(1)

		# X_in = normalize(X_in)
		# X_mix = normalize(X_mix)
		c = adapt_model.train(X_mix, X_in,learning_rate, i)
		# c = adapt_model.test(X_mix, X_in)
		# if (i+1)%100==0:
		# 	learning_rate /= 10

		print 'Step #'  ,i,' ', c 

		if i%20 == 0: #cost_valid < cost_valid_min:
			print 'DAS model saved at iteration number ', i,' with cost = ', c 
			# cost_valid_min = cost_valid
			adapt_model.save(i)
			last_saved = i

		# 	if i - last_saved > config.stop_iterations:
		# 		print 'Stop'
		# 		break

		# 	