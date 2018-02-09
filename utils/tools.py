#coding: utf-8
from __future__ import print_function 
import sys
# Returns the string of remaining training time
def getETA(batchTime, nbBatch, batchIndex, nbEpoch, epoch):
	seconds = int(batchTime*(nbBatch-batchIndex-1) + batchTime*nbBatch*(nbEpoch-epoch-1))
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	return "%dh%02dm%02ds" % (h, m, s)


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		bar_length  - Optional  : character length of bar (Int)
	"""
	str_format = "{0:." + str(decimals) + "f}"
	percents = str_format.format(100 * (iteration / float(total)))
	filled_length = int(round(bar_length * iteration / float(total)))
	bar = '█' * filled_length + '-' * (bar_length - filled_length)

	sys.stdout.write('\r{0} |{1}| {2}{3} {4}\r'.format(prefix, bar, percents, '%', suffix)),

	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()


def args_to_string(args):
	items = []
	for key, val in sorted(args.items()):
		k = key[:5]
		v = val
		if v == True:
			v = 'T'
		elif v == False:
			v = 'F'
		elif isinstance(v, (str, unicode)):
			if '/' in v:
				split = v.split('/')
				v = split[1][:-3]
		items.append((k,v))

	return ''.join('_{}_{}'.format(k, v) for k, v in items)

import numpy as np

def normalize_mix(X_mix, X_non_mix, type='min-max'):
	if type == 'min-max':
		a = 0.0
		b = 1.0
		max_val = np.amax(X_mix, axis=-1, keepdims=True)
		min_val = np.amin(X_mix, axis=-1, keepdims=True)
		S = float(X_non_mix.shape[1])
		A = (b - a)/(max_val - min_val)
		B = b - A * max_val
		X_mix = A*X_mix + B
		X_non_mix = A[:,:,np.newaxis]*X_non_mix + B[:,:,np.newaxis] / S

	return X_mix, X_non_mix
