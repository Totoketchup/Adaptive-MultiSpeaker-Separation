import tensorflow as tf 
import os 
import config
from utils.postprocessing.representation import *
import matplotlib.pyplot as plt
import numpy as np
# from models import Adapt

## Here we plot the windows and bases of the Adapt model

# /home/anthony/das/log/pretraining/
# AdaptiveNet-noisy-breeze-3898-N=256--
# alpha=0.01--batch_size=16--beta=0.05--
# chunk_size=20480--maxpool=256--optimizer=Adam--
# reg=0.001--rho=0.01--same_filter=True--
# smooth_size=10--type=pretraining--window=1024-/

####
#### MODEL CONFIG
####

full_id = "AdaptiveNet-aged-firefly-5068-N=16--alpha=0.01--batch_size=16--beta=0.05--chunk_size=20480--maxpool=4--optimizer=Adam--reg=0.0001--rho=0.01--same_filter=True--smooth_size=10--type=pretraining--window=1024-"
model_meta_path = os.path.join(config.log_dir, "pretraining", full_id, 'model.ckpt.meta')
model_path = os.path.join(config.log_dir, "pretraining", full_id, 'model.ckpt')

sess = tf.Session()

importer = tf.train.import_meta_graph(model_meta_path)
importer.restore(sess, model_path)

graph = tf.get_default_graph()

window_front = graph.get_tensor_by_name('front/window/w:0')
bases_front = graph.get_tensor_by_name('front/bases/bases:0')

window_back = graph.get_tensor_by_name('back/window_2/w_2:0')
bases_back = graph.get_tensor_by_name('back/bases_2/bases_2:0')


with sess.as_default():
	window_front = window_front.eval()
	bases_front = np.transpose(bases_front.eval())
	window_back = window_back.eval()
	bases_back = np.transpose(bases_back.eval())

plt.plot(np.abs(window_front), label='front window')
plt.show()

print bases_front.shape

tot = 100

for i in range(10):
	plt.plot(bases_front[i])
	plt.show()
	# plt.subplot(10,10,i+1)

plt.show()

