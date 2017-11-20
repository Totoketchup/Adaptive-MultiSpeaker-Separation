# coding: utf-8

from data.dataset import H5PY_RW
from data.data_tools import read_metadata, males_keys, females_keys
from data.dataset import Mixer
from models.adapt import Adapt
from utils.tools import getETA
import time
import numpy as np

H5_dic = read_metadata()
chunk_size = 512*40

males = H5PY_RW('test_raw.h5py', subset = males_keys(H5_dic))
fem = H5PY_RW('test_raw.h5py', subset = females_keys(H5_dic))

print 'Data with', len(H5_dic), 'male and female speakers'
print males.length(), 'elements'
print fem.length(), 'elements'

mixed_data = Mixer([males, fem], chunk_size= chunk_size, with_mask=False, with_inputs=True, shuffling=True)


####
####

N = 256
max_pool = 256
batch_size = 16
learning_rate = 0.01

config_model = {}
config_model["type"] = "pretraining"

config_model["batch_size"] = batch_size
config_model["chunk_size"] = chunk_size

config_model["N"] = N
config_model["maxpool"] = max_pool
config_model["window"] = 1024

config_model["smooth_size"] = 10

config_model["alpha"] = learning_rate
config_model["reg"] = 1e-3
config_model["beta"] = 0.05
config_model["rho"] = 0.01

config_model["same_filter"] = True
config_model["optimizer"] = 'Adam'

####
####

adapt_model = Adapt(config_model=config_model, pretraining=True, folder='pretraining')
adapt_model.tensorboard_init()
adapt_model.init()

print 'Total name :' 
print adapt_model.runID

# nb_iterations = 500
mixed_data.adjust_split_size_to_batchsize(batch_size)
nb_batches = mixed_data.nb_batches(batch_size)
nb_epochs = 10

time_spent = [ 0 for _ in range(5)]

for epoch in range(nb_epochs):
	for b in range(nb_batches):
		step = nb_batches*epoch + b
		X_non_mix, X_mix, _ = mixed_data.get_batch(batch_size)
		t = time.time()
		c = adapt_model.train(X_mix, X_non_mix, learning_rate, step)
		t_f = time.time()
		time_spent = time_spent[1:] +[t_f-t]

		print 'Step #'  , step,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
			, nb_batches, b, nb_epochs, epoch)

		if b%20 == 0:
		    print 'DAS model saved at iteration number ', step,' with cost = ', c 
		    adapt_model.save(b)

