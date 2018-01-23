# coding: utf-8

from data.dataset import H5PY_RW
from data.data_tools import read_metadata, males_keys, females_keys
from data.dataset import Mixer
from models.adapt import Adapt
from utils.tools import getETA
import time
import numpy as np

H5_dic = read_metadata()

males = H5PY_RW('dev-clean.h5', subset = males_keys(H5_dic))
fem = H5PY_RW('dev-clean.h5', subset = females_keys(H5_dic))

print 'Data with', len(H5_dic), 'male and female speakers'
print males.length(), 'elements'
print fem.length(), 'elements'

chunk_size = 512*20
S = 2
mixed_data = Mixer([males, fem], chunk_size= chunk_size, 
	with_mask=False, with_inputs=True, shuffling=True,
	nb_speakers=S, random_picking=True)

####

batch_size = 1

config_model = {
	"type" : "pretraining",
	"batch_size" : batch_size,
	"chunk_size" : chunk_size,
	"N" : 516,
	"maxpool" : 256,
	"window" : 1048,
	"alpha" : 0.01,
	"reg" : 1e-4,
	"beta" : 5e-4,
	"rho" : 0.1,
	"optimizer" : "Adam",
	"speakers" : S
}

####

adapt_model = Adapt(config_model=config_model, pretraining=True, folder='pretraining')
adapt_model.tensorboard_init()
adapt_model.init()

print 'Total name :' 
print adapt_model.runID

# nb_iterations = 500
mixed_data.adjust_split_size_to_batchsize(batch_size)
nb_batches = mixed_data.nb_batches(batch_size)
nb_epochs = 30

time_spent = [ 0 for _ in range(5)]

for epoch in range(nb_epochs):
	for b in range(nb_batches):
		step = nb_batches*epoch + b
		X_non_mix, X_mix, _ = mixed_data.get_batch(batch_size)
		mean = np.mean(X_mix)
		std = np.std(X_mix)
		X_mix = (X_mix - mean)/std
		X_non_mix = (X_non_mix)/std

		t = time.time()
		c = adapt_model.train(X_mix, X_non_mix, config_model["alpha"], step)
		t_f = time.time()
		time_spent = time_spent[1:] +[t_f-t]

		print 'Step #'  , step,' loss=', c ,' ETA = ', getETA(sum(time_spent)/float(np.count_nonzero(time_spent))
			, nb_batches, b, nb_epochs, epoch)

		if b%20 == 0:
		    print 'L41 FRONT model saved at iteration number ', step,' with cost = ', c 
		    adapt_model.save(b)

