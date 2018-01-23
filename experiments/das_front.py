# coding: utf-8

from data.dataset import H5PY_RW
from data.data_tools import read_metadata, males_keys, females_keys
from data.dataset import Mixer
from models.adapt import Adapt
from models.das import DAS
from utils.tools import getETA
import time
import numpy as np
import config
import os

H5_dic = read_metadata()
chunk_size = 512*40

males = H5PY_RW('test_raw.h5py', subset = males_keys(H5_dic))
fem = H5PY_RW('test_raw.h5py', subset = females_keys(H5_dic))

print 'Data with', len(H5_dic), 'male and female speakers'
print males.length(), 'elements'
print fem.length(), 'elements'

mixed_data = Mixer([males, fem], chunk_size= chunk_size, with_mask=False, with_inputs=True)


####
#### PREVIOUS MODEL CONFIG
####

N = 256
max_pool = 256
batch_size = 1
learning_rate = 0.001

config_model = {}
config_model["type"] = "pretraining"

config_model["batch_size"] = batch_size
config_model["chunk_size"] = 512*40

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
#### LOAD PREVIOUS MODEL
####

idd = ''.join('-{}={}-'.format(key, val) for key, val in sorted(config_model.items()))
config_model["type"] = "DAS_train_front"
learning_rate = 0.001
batch_size = 32
config_model["chunk_size"] = chunk_size
config_model["batch_size"] = batch_size
config_model["alpha"] = learning_rate
config_model["optimizer"] = 'RMS'

full_id = "restless-dew-7622" + idd


folder = 'DAS_train_front'
model = Adapt(config_model=config_model, pretraining=False)
model.create_saver()

path = os.path.join(config.model_root, 'log', 'pretraining')
model.restore_model(path, full_id)

model.connect_only_front_to_separator(DAS)
init = model.non_initialized_variables()

model.sess.run(init)

print 'Total name :' 
print model.runID

# nb_iterations = 500
mixed_data.adjust_split_size_to_batchsize(batch_size)
nb_batches = mixed_data.nb_batches(batch_size)
nb_epochs = 1

time_spent = [ 0 for _ in range(5)]

for epoch in range(nb_epochs):
	for b in range(nb_batches):
		X_non_mix, X_mix, Ind = mixed_data.get_batch(batch_size)
		t = time.time()
		c = model.train(X_mix, X_non_mix, learning_rate, b, ind_train=Ind)
		t_f = time.time()
		time_spent = time_spent[1:] +[t_f-t]

		print 'Step #'  ,b,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
			, nb_batches, b, nb_epochs, epoch)
		# print 'length of data =', X_non_mix.shape ,'step ', b+1, mixed_data.datasets[0].index_item_split, mixed_data.selected_split_size(),getETA(sum(time_spent)/float(np.count_nonzero(time_spent)), nb_batches, b, nb_epochs, epoch)

		if b%20 == 0: #cost_valid < cost_valid_min:
			print 'DAS model saved at iteration number ', nb_batches*epoch + b,' with cost = ', c 
			model.save(b)

