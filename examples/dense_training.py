import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Get your datasets

from data.dataset import H5PY_RW
from data.data_tools import read_data_header, males_keys, females_keys

file = 'test_raw_16k.h5py'
H5_dic = read_data_header()
chunk_size = 512*10

males = H5PY_RW().open_h5_dataset(file, subset = males_keys(H5_dic)).set_chunk(chunk_size).shuffle()
fem = H5PY_RW().open_h5_dataset(file, subset = females_keys(H5_dic)).set_chunk(chunk_size).shuffle()
print 'Data with', len(H5_dic), 'male and female speakers'


# Mixing the dataset

from data.dataset import Mixer

mixed_data = Mixer([males, fem], with_mask=False, with_inputs=True)

# Training set selection
mixed_data.select_split(0)

# Model pretrained loading

N = 256
max_pool = 128
batch_size = 8
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

idd = ''.join('-{}={}-'.format(key, val) for key, val in sorted(config_model.items()))
batch_size = 4
config_model["batch_size"] = batch_size
config_model["type"] = "Dense_train"

from models.adapt import Adapt
import config

full_id = 'soft-base-9900'+idd

folder='Dense_train'
model = Adapt(config_model=config_model,pretraining=False)
model.create_saver()

path = os.path.join(config.workdir, 'floydhub_model', "pretraining")
# path = os.path.join(config.log_dir, "pretraining")
model.restore_model(path, full_id)

## Connect DAS model to the front end

from models.dense import Dense_net as Dense

with model.graph.as_default():
	model.connect_front(Dense)
	model.sepNet.output = model.sepNet.prediction
	model.back
	model.cost
	model.optimize
	# model.freeze_front()
	# model.optimize
	model.tensorboard_init()

from itertools import compress
with model.graph.as_default():
	global_vars = tf.global_variables()
	is_not_initialized = model.sess.run([~(tf.is_variable_initialized(var)) \
								   for var in global_vars])
	not_initialized_vars = list(compress(global_vars, is_not_initialized))
	if len(not_initialized_vars):
		init = tf.variables_initializer(not_initialized_vars)


# Model creation

# Pretraining the model 

nb_iterations = 1000

#initialize the model
model.sess.run(init)

for i in range(nb_iterations):
	X_in, X_mix, Ind = mixed_data.get_batch(batch_size)
	c = model.train(X_mix, X_in,learning_rate, i, ind_train=Ind)
	print 'Step #'  ,i,' loss=', c 

	if i%20 == 0: #cost_valid < cost_valid_min:
		print 'DAS model saved at iteration number ', i,' with cost = ', c 
		model.save(i)