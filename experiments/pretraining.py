# coding: utf-8

from data.dataset import H5PY_RW
from data.data_tools import read_metadata, males_keys, females_keys
from data.dataset import Mixer
from models.adapt import Adapt
from utils.tools import getETA
import time
import numpy as np
H5_dic = read_metadata()
chunk_size = 512*10

males = H5PY_RW('test_raw.h5py', subset = males_keys(H5_dic)).set_chunk(chunk_size).shuffle()
fem = H5PY_RW('test_raw.h5py', subset = females_keys(H5_dic)).set_chunk(chunk_size).shuffle()

print 'Data with', len(H5_dic), 'male and female speakers'
print males.length(), 'elements'
print fem.length(), 'elements'

mixed_data = Mixer([males, fem], with_mask=False, with_inputs=True)


####
####

N = 512
max_pool = 256
batch_size = 1
learning_rate = 0.001

config_model = {}
config_model["type"] = "pretraining"

config_model["batch_size"] = batch_size
config_model["chunk_size"] = chunk_size

config_model["N"] = N
config_model["maxpool"] = max_pool
config_model["window"] = 1024

config_model["smooth_size"] = 20

config_model["alpha"] = learning_rate
config_model["reg"] = 1e-4
config_model["beta"] = 0.1
config_model["rho"] = 0.01

config_model["same_filter"] = False
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
nb_epochs = 1

time_spent = [ 0. for _ in range(3)]


for epoch in range(nb_epochs):
	for b in range(nb_batches):
		t = time.time()
		X_non_mix, _, _ = mixed_data.get_batch(batch_size)

		c = adapt_model.pretrain(X_non_mix, learning_rate, b)
		t_f = time.time()
		time_spent = time_spent[1:] +[t_f-t]

		print 'Step #'  ,b,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
			, nb_batches, b, nb_epochs, epoch)
		# print b+1, mixed_data.datasets[0].index_item_split, mixed_data.selected_split_size()

		if b%20 == 0: #cost_valid < cost_valid_min:
		    print 'DAS model saved at iteration number ', nb_batches*epoch + b,' with cost = ', c 
		    adapt_model.save(b)

