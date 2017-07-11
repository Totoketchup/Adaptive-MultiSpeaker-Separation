import numpy as np
from ..audio import istft

# Compute the reconstruction of the signal from the filtered spectogram
def reconstruct_signal(filtered_spec, orig_spec, fs=config.fs, fftsize=config.fftsize):
	
	if orig_spec != None :
		angle = np.angle(orig_spec)
		phi = np.exp(1.0j*np.unwrap(angle)) # the angle must be 2pi modulo
	else
		phi = np.random.randn(*orig_spec.shape)

	return istft(filtered_spec*phi)
