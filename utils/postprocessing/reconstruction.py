import numpy as np
from audio import istft_, create_spectrogram
from sklearn.cluster import KMeans

# Compute the reconstruction of the signal from the filtered spectrogram
def reconstruct_signal(filtered_spec, orig_spec, fs=config.fs, fftsize=config.fftsize):
	
	if orig_spec != None :
		angle = np.angle(orig_spec)
		phi = np.exp(1.0j*np.unwrap(angle)) # the angle must be 2pi modulo
	else
		phi = np.random.randn(*orig_spec.shape)

	return istft_(filtered_spec*phi)


# Produce the embeddings of a mixture spectogram through a Deep Learning modle
def produce_embeddings(model, X):
	# If there is only 1 mixture (no batch)
	if len(X.shape) == 2:
		X = np.reshape(X, (1, X.shape[0], X.shape[1]))

	return model.embeddings(X)

def produce_masks(embeddings, nb_speakers):

	Kmean = KMeans(n_clusters = nb_speakers, n_jobs = -1)

	Kmean.fit(np.reshape(embeddings, (-1, embeddings.shape[2])))

	masks = [np.eye(nb_speakers)[label] for label in Kmean.labels_]

	masks = np.reshape(masks, embeddings.shape)

	return masks

def apply_masks(X, masks):
	return [masks[:, :, i]*X for i in range(masks.shape[2])]

def separate(signal, sample_rate, nb_speakers, model):

	spectrogram = create_spectrogram(signal, sample_rate)

	X = np.sqrt(abs(spectrogram))

	min_ = X.min()
	max_ = X.max()
	X = (X - min_)/(max_ - min_) # Normalized absolute spectrogram

	#Pass the normalized spec into the model
	embeddings = produce_embeddings(model, X)

	masks = produce_masks(embeddings, nb_speakers)

	filtered_specs = apply_masks(X, masks)

	signals = [reconstruct_signal(spec, spectrogram) for spec in filtered_specs]

	return signals