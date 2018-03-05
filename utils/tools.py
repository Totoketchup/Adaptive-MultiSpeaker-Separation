#coding: utf-8
from __future__ import print_function 

# Returns the string of remaining training time
def getETA(batchTime, nbBatch, batchIndex, nbEpoch, epoch):
	seconds = int(batchTime*(nbBatch-batchIndex-1) + batchTime*nbBatch*(nbEpoch-epoch-1))
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	return "%dh%02dm%02ds" % (h, m, s)

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
