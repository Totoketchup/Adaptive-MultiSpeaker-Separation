from scipy.signal import stft,istft,resample
import numpy as np
from subprocess import call
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import config

def stft_(x, fs=config.fs, fftsize=config.fftsize, overlap=config.fftsize//config.overlap, window='hann', padded=True):
    f, t , X = stft(x, fs, window, fftsize, overlap, padded=padded)
    return f, t, X

def spectrogram(stft):
    X = np.log10(np.absolute(stft))
    return X

def istft_(X, fs=config.fs, fftsize=config.fftsize, overlap=config.fftsize//config.overlap, window='hann'):
    t, x_recons = istft(X, fs, window, fftsize , overlap)
    return t, x_recons

def binary_mask(X1, X2):
	return np.array([(X1 >= X2).astype(int), (X2 > X1).astype(int)])

def downsample(audio_data, sample_rate, new_sample_rate):
    return resample(audio_data, int(len(audio_data)/float(sample_rate)*new_sample_rate))

def create_spectrogram(x, sr, fs=config.fs, fftsize=config.fftsize, overlap=config.fftsize//config.overlap, window=config.window, padded=True):
    x = downsample(x, sr, fs) # Downsampling the original signal
    return stft_(x, fs, fftsize, overlap, window, padded) # Compute the STFT of the downsampled signal
