import tensorflow as tf 
import os 
import config
from utils.postprocessing.representation import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
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
def main(args):

	sess = tf.Session()

	checkpoint =  tf.train.latest_checkpoint(args.path)

	importer = tf.train.import_meta_graph(checkpoint+'.meta')
	importer.restore(sess, checkpoint)

	graph = tf.get_default_graph()

	front_window = graph.get_tensor_by_name('front/window/w:0')
	front_bases = graph.get_tensor_by_name('front/bases/bases:0')

	with sess.as_default():
		front_window = front_window.eval()
		front_bases = front_bases.eval()
		front_bases = np.transpose(front_bases)

	sub = 256 / 16

	for j in range(sub):
	    fig, plots = plt.subplots(4, 4, figsize=(18, 16))
	    
	    for x in range(4):
	        for y in range(4):
	            plots[x, y].plot(front_window*front_bases[j*16+(4*y+x)])
	            plots[x, y].axis([0,1024,-0.01,0.01])
	    plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Argument Parser")

	parser.add_argument(
		'--path', help='Path to Adapt model', required=True)

	main(parser.parse_args())
			