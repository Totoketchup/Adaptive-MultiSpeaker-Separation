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
batch_size = 64
learning_rate = 0.01

config_model = {}
config_model["type"] = "L41_train_front"

config_model["batch_size"] = batch_size
config_model["chunk_size"] = 512*10

config_model["N"] = N
config_model["maxpool"] = max_pool
config_model["window"] = 1024

config_model["smooth_size"] = 10

config_model["alpha"] = learning_rate
config_model["reg"] = 1e-3
config_model["beta"] = 0.1
config_model["rho"] = 0.01

config_model["same_filter"] = True
config_model["optimizer"] = 'Adam'


idd = ''.join('-{}={}-'.format(key, val) for key, val in sorted(config_model.items()))
# full_id = 'green-sound-9629'+idd
full_id ='purple-term-1311' + idd

####
#### NEW MODEL CONFIGURATION
####

config_model["type"] = "L41_finetuning"
learning_rate = 0.001 
batch_size = 1
config_model["chunk_size"] = chunk_size
config_model["alpha"] = learning_rate
config_model["batch_size"] = batch_size

model = Adapt(config_model=config_model, pretraining=False)
model.create_saver()

path = os.path.join(config.model_root, 'log', 'L41_train_front')
model.restore_model(path, full_id)

model.connect_front_back_to_separator(L41Model)

with model.graph.as_default():
    model.create_saver()
    model.restore_model(path, full_id)
    # model.freeze_front()
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
		X_non_mix, X_mix, _ = mixed_data.get_batch(batch_size)
		t = time.time()
		c = model.train(X_mix, X_non_mix, learning_rate, step)
		t_f = time.time()
		time_spent = time_spent[1:] +[t_f-t]

		print 'Step #'  ,step,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
			, nb_batches, b, nb_epochs, epoch)
		# print 'length of data =', X_non_mix.shape ,'step ', b+1, mixed_data.datasets[0].index_item_split, mixed_data.selected_split_size(),getETA(sum(time_spent)/float(np.count_nonzero(time_spent)), nb_batches, b, nb_epochs, epoch)

		if b%20 == 0: #cost_valid < cost_valid_min:
		    print 'DAS model saved at iteration number ',step,' with cost = ', c 
		    model.save(b)

model.savedModel()