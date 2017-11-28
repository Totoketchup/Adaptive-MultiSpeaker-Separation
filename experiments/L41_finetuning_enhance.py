# coding: utf-8

from data.dataset import H5PY_RW
from data.data_tools import read_metadata, males_keys, females_keys
from data.dataset import Mixer
from models.adapt import Adapt
from models.L41 import L41Model
from utils.tools import getETA
import time
import numpy as np
import config
import os
import tensorflow as tf
H5_dic = read_metadata()
chunk_size = 512*10

males = H5PY_RW('test_raw.h5py', subset = males_keys(H5_dic))
fem = H5PY_RW('test_raw.h5py', subset = females_keys(H5_dic))

print 'Data with', len(H5_dic), 'male and female speakers'
print males.length(), 'elements'
print fem.length(), 'elements'

mixed_data = Mixer([males, fem], chunk_size= chunk_size, with_mask=False, with_inputs=True, shuffling=True)

####
#### PREVIOUS ADAPT MODEL CONFIG
####

N = 256
max_pool = 256
batch_size = 16
learning_rate = 0.01

adapt_model = {}
adapt_model["type"] = "pretraining"

adapt_model["batch_size"] = batch_size
adapt_model["chunk_size"] = 512*40

adapt_model["N"] = N
adapt_model["maxpool"] = max_pool
adapt_model["window"] = 1024

adapt_model["smooth_size"] = 10

adapt_model["alpha"] = learning_rate
adapt_model["reg"] = 1e-3
adapt_model["beta"] = 0.05
adapt_model["rho"] = 0.01

adapt_model["same_filter"] = True
adapt_model["optimizer"] = 'Adam'
idd_adapt = ''.join('-{}={}-'.format(key, val) for key, val in sorted(adapt_model.items()))
full_id_adapt = "noisy-breeze-3898" + idd_adapt
path_adapt = os.path.join(config.model_root if not config.floydhub else '/model1', 'log', 'pretraining')

####
#### PREVIOUS MODEL CONFIG
####


N = 256
max_pool = 256
batch_size = 64
learning_rate = 0.001

config_model = {}
config_model["type"] = "L41_enhance"

config_model["batch_size"] = batch_size
config_model["chunk_size"] = 512*10

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



idd = ''.join('-{}={}-'.format(key, val) for key, val in sorted(config_model.items()))
full_id ='cold-dust-9076' + idd
path = os.path.join(config.model_root if not config.floydhub else '/model2', 'log', config_model["type"] )

####
#### NEW MODEL CONFIGURATION
####

config_model["type"] = "L41_finetuning"
learning_rate = 0.001 
batch_size = 64
config_model["chunk_size"] = chunk_size
config_model["alpha"] = learning_rate
config_model["batch_size"] = batch_size

model = Adapt(config_model=config_model, pretraining=False)



with model.graph.as_default():
	model.connect_front(L41Model)
	var_list =[v for v in tf.global_variables() if ('front' in v.name)]
	model.create_saver(var_list)
	model.restore_model(path_adapt, full_id_adapt)
	model.sepNet.prediction
	model.sepNet.separate
	model.sepNet.output = model.sepNet.enhance
	var_list =[v for v in tf.global_variables() if ('prediction' in v.name or 'speaker_centroids' in v.name or 'enhance' in v.name)]
	model.create_saver(var_list)
	model.restore_model(path, full_id)
	model.separator
	model.back
	var_list = [v for v in tf.global_variables() if ('back/' in v.name)]
	model.create_saver(var_list)
	model.restore_model(path_adapt, full_id_adapt)
	model.cost
	model.freeze_variables()
	model.optimize
	model.tensorboard_init()

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
		step = nb_batches*epoch + b
		X_non_mix, X_mix, Ind = mixed_data.get_batch(batch_size)
		t = time.time()
		c = model.train_no_sum(X_mix, X_non_mix,learning_rate, step, ind_train= Ind)
		t_f = time.time()
		time_spent = time_spent[1:] +[t_f-t]

		print 'Step #'  ,step,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
			, nb_batches, b, nb_epochs, epoch)
		# print 'length of data =', X_non_mix.shape ,'step ', b+1, mixed_data.datasets[0].index_item_split, mixed_data.selected_split_size(),getETA(sum(time_spent)/float(np.count_nonzero(time_spent)), nb_batches, b, nb_epochs, epoch)

		if b%20 == 0: #cost_valid < cost_valid_min:
		    print 'DAS model saved at iteration number ',step,' with cost = ', c 
		    model.save(b)

model.savedModel()