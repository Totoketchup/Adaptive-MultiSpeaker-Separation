# -*- coding: utf-8 -*-
import h5py
import numpy as np
import data_tools
import soundfile as sf
import os
import config
from utils.audio import create_spectrogram, downsample
from sets import Set
from utils.tools import print_progress
from tqdm import tqdm 

class H5PY_RW:

	def __init__(self, filename, subset=None):
		"""
		Open a LibriSpeech H5PY file.
		
		Inputs:
			filename: name of h5 file
			subset: subset of speakers used, default = None (all in the file)
		"""

		path = os.path.join(config.h5py_root, filename)
		self.h5 = h5py.File(path, 'r')
		
		# Define the the keys for each speaker
		if subset == None:
			self.keys = [key for key in self.h5] # All
		else:
			self.keys = subset # Subset of the H5 file


		# Define all the items related to each key/speaker
		items = []
		for key in self.keys:
			items += [key + '/' + val  for val in self.h5[key]]

		self.raw_items = items # Original definition of items, never modified
		self.items = items # current definition of items, modified according to 'chunk' parameter
		self.index_item = 0 # current index 

		self.chunk_size = 0 # If = 0 then no chunk applied, all the audio is used

	def set_chunk(self, chunk_size):
		"""
		Define the size of the chunk: each data is chunked with 'chunk_size'
		"""

		self.chunk_size = chunk_size
		items = []

		# Chunk along the first axis
		# Define items like this:
		# items = ['speaker_key/audio_file/part_according_to_chunk']
		# For example, with 3 parts:
		# items = ['speaker_key/audio_file/0', 'speaker_key/audio_file/1', 'speaker_key/audio_file/2']
		for item in self.raw_items:
			L = self.h5[item].shape[0]//chunk_size
			items += [item +'/' + str(part) for part in range(L)]

		# Update the items into chunked items
		self.items = items
		return self

	def next_in_split(self, splits, split_index):
		"""
		Return next chunked item in the indicated split
		Input:
			splits: ratio of each split (array)
			index: split index
		"""
		if not hasattr(self, 'index_item_split'):
			self.index_item_split = np.array([ int(sum(splits[0:i])*len(self.items)) for i in range(len(splits))], dtype = np.int32)
		# print self.index_item_split
		item_path = self.items[self.index_item_split[split_index]]
		split_path = item_path.split('/')

		# Speaker indice
		key = int(split_path[0])

		if self.chunk_size !=0:
			# Which part to chunk
			i = int(split_path[2])

			# Cut the data according to the chunk
			item_path = '/'.join(split_path[:2])
			X = self.h5[item_path][i*self.chunk_size : (i+1)*self.chunk_size]
		else:
		 	X = self.h5[item_path][:] # get the full data

		self.index_item_split[split_index]+=1
		if self.index_item_split[split_index] >= int(sum(splits[0:split_index+1])*len(self.items)):
			self.index_item_split[split_index] = int(sum(splits[0:split_index])*len(self.items))

		return X, key

	def reset_split(self, splits, index):
		self.index_item_split[index] = int(sum(splits[0:index])*len(self.items))

	def shuffle(self, seed=1):
		np.random.seed(seed)
		np.random.shuffle(self.items)
		return self

	def speakers(self):
		return self.keys

	def length(self):
		return len(self.items)

	def __iter__(self):
		return self





	@staticmethod
	def create_h5_dataset(self, output_fn, subset=config.data_subset, data_root=config.data_root):
		"""
		Create a H5 file from the LibriSpeech dataset and the subset given:

		Inputs:
			output_fn: filename for the created file
			subset: LibriSpeech subset : 'dev-clean' , ...
			data_root: LibriSpeech folder path

		"""

		# Extract the information about this subset (speakers, chapters)
		# Dictionary with the following shape: 
		# {speaker_key: {chapters: [...], sex:'M/F', ... } }
		speakers_info = data_tools.read_metadata(subset)

		with h5py.File(output_fn,'w') as data_file:

			for (key, elements) in speakers_info.items():
				if key not in data_file:
					# Create an H5 Group for each key/speaker
					data_file.create_group(key)

				# Current speaker folder path
				folder = data_root+'/'+subset+'/'+key

				print_progress(0, len(elements['chapters']), prefix = 'Speaker '+key+' :', suffix = 'Complete')

				# For all the chapters read by this speaker
				for i, chapter in enumerate(elements['chapters']): 
					# Find all .flac audio
					for root, dirs, files in os.walk(folder+'/'+chapter): 
						for file in files:
							if file.endswith(".flac"):

								path = os.path.join(root,file)
								raw_audio, samplerate = sf.read(path)

								# Generate the spectrogram for the current audio file
								_, _, spec = create_spectrogram(raw_audio, samplerate)

								data_file[key].create_dataset(file,
									data=spec.T.astype(np.complex64),
									compression="gzip",
									dtype=np.complex64,
									compression_opts=0)

					print_progress(i + 1, len(elements['chapters']), prefix = 'Speaker '+key+' :', suffix = 'Complete')


		print 'Dataset for the subset: ' + subset + ' has been built'

	@staticmethod
	def create_raw_audio_dataset(output_fn, subset=config.data_subset, data_root=config.data_root):
		"""
		Create a H5 file from the LibriSpeech dataset and the subset given:

		Inputs:
			output_fn: filename for the created file
			subset: LibriSpeech subset : 'dev-clean' , ...
			data_root: LibriSpeech folder path

		"""
		from librosa.core import resample,load

		# Extract the information about this subset (speakers, chapters)
		# Dictionary with the following shape: 
		# {speaker_key: {chapters: [...], sex:'M/F', ... } }
		speakers_info = data_tools.read_metadata(subset)
		with h5py.File(output_fn,'w') as data_file:

			for key, elements in tqdm(speakers_info.items(), total=len(speakers_info), desc='Speakers'):
				if key not in data_file:
					# Create an H5 Group for each key/speaker
					data_file.create_group(key)

				# Current speaker folder path
				folder = data_root+'/'+subset+'/'+key
				# For all the chapters read by this speaker
				for i, chapter in enumerate(tqdm(elements['chapters'], desc='Chapters')): 
					# Find all .flac audio
					for root, dirs, files in os.walk(folder+'/'+chapter): 
						for file in tqdm(files, desc='Files'):
							if file.endswith(".flac"):
								path = os.path.join(root,file)
								raw_audio, sr = load(path, sr=16000)
								raw_audio = resample(raw_audio, sr, config.fs)
								data_file[key].create_dataset(file,
									shape=raw_audio.shape,
									data=raw_audio,
									chunks=raw_audio.shape,
									maxshape=raw_audio.shape,
									compression="gzip",
									compression_opts=9)

		print 'Dataset for the subset: ' + subset + ' has been built'





class Mixer:

	def __init__(self, datasets, chunk_size=0, shuffling=False, with_mask=True, 
		with_inputs=False, splits = [0.8, 0.1, 0.1], mixing_type='add', mask_positive_value=1, 
		mask_negative_value=-1, nb_speakers = 2, random_picking=True):
		"""
		Mix multiple H5PY file writer/reader (H5PY_RW)
		Inputs:
			datasets: array of H5PY reader
			mixing_type: 'add' (Default), 'mean'
			splits: Percentage for Training Set (Default 0.8), for Testing set (Default 0.1), for Validation Set (Default 0.1)
			mask_positive_value: Bin value if the spectrogram bin belong to the mask
			mask_negative_value: Bin value if the spectrogram bin does not belong to the mask
		"""
		self.datasets = datasets
		self.type = mixing_type
		self.create_labels()
		self.with_mask = with_mask
		self.with_inputs = with_inputs
		self.mixing_type = mixing_type
		self.mask_negative_value = mask_negative_value
		self.mask_positive_value = mask_negative_value
		self.splits = splits
		self.split_index = 0 # Training split by default
		self.chunk_size = chunk_size
		self.nb_speakers = nb_speakers
		self.random_picking = random_picking

		if chunk_size !=0:
			for dataset in datasets:
				dataset.set_chunk(chunk_size)

		if shuffling:
			self.shuffle()

		# Align both dataset:
		self.size = np.amin([x.length() for x in datasets])
		for dataset in datasets:
			# Adjust size to the smallest dataset in order to have 
			# regular epochs
			dataset.items = dataset.items[0:self.size]

		self.splits_size = [int(split_ratio*self.size) for split_ratio in splits]

	def shuffle(self):
		for i, dataset in enumerate(self.datasets):
			dataset.shuffle(i+1)

	def select_split(self, index):
		self.split_index = index
		return self

	def adjust_split_size_to_batchsize(self, batch_size):
		self.batch_size = batch_size
		self.size -= self.size%int(batch_size/self.splits[self.split_index])
		for dataset in self.datasets:
			# Adjust size to the batchsize for this split
			dataset.items = dataset.items[0:self.size]
		print 'size =' ,self.size

	def selected_split_size(self):
		return int(self.splits[self.split_index]*self.size)

	def nb_batches(self, batch_size):
		return int(self.splits[self.split_index]*self.size/batch_size)

	def next(self):
		X_d = []
		key_d = []

		# Where dataset are chosen:
		if self.random_picking:
			# Take the number of desired speakers randomly among the datasets [Males, Females]
			choice = np.random.choice(len(self.datasets), self.nb_speakers, replace=True)
		else:
			# Take [M, F, M, F ... ] 'nb_speakers times'
			choice = []
			indicies = np.arange(len(self.datasets))
			choice = np.array([indicies[i%len(indicies)] for i in range(self.nb_speakers)])

		for dataset in np.array(self.datasets)[choice]:
			X, key = dataset.next_in_split(self.splits, self.split_index)
			X_d.append(X)
			key_d.append(key)

		if self.datasets[0].chunk_size == 0: # If there is no chunk but the full data
			# Adjust the size of each input to the smallest one 
			lengths = [len(x) for x in X_d]
			min_ = np.amin(lengths)
			X_d = [x[:min_] for x in X_d]

		key_d = np.array(key_d)
		X_non_mix = np.array(X_d)


		if self.mixing_type == 'add':
			X_mix = np.sum(X_non_mix, axis=0)
		elif self.mixing_type == 'mean':
			X_mix = np.mean(X_non_mix, axis=0)

		if self.with_mask:
			Y_pos = np.argmax(X_non_mix, axis=0)
			Y_pos[Y_pos == 0 ] = self.mask_negative_value
			Y = np.array([Y_pos, -Y_pos]).transpose((1,2,0))

		if self.with_inputs:
			return X_non_mix, X_mix, key_d
		elif self.with_inputs and self.with_mask:
			return X_mix, Y, key_d, X_non_mix
		else:
			return X_mix, key_d

	def get_batch(self, batch_size):
		X_mix = []
		Ind = []

		if self.with_mask:
			Y = []
		if self.with_inputs:
			X_non_mix = []


		for i in range(batch_size):
			if self.with_mask:
				x_mix, y, ind = self.next()
				Y.append(y)
			elif self.with_inputs:
				x_non_mix, x_mix, ind = self.next()
				X_non_mix.append(x_non_mix)

			X_mix.append(x_mix)
			Ind.append([self.dico[j] for j in ind])

		if self.with_mask:
			return np.array(X_mix), np.array(Y), np.array(Ind)

		if self.with_inputs:
			return np.array(X_non_mix), np.array(X_mix), np.array(Ind)

		return np.array(X_mix), np.array(Ind)

	def get_only_first_items(self, size):
		batch = self.get_batch(size)
		# Hack on the split index -> reset to zero
		for dataset in self.datasets:
			dataset.reset_split(self.splits, self.split_index)
		return batch

	def create_labels(self):
		# Create a set of all the speaker indicies 
		self.items = Set()
		for dataset in self.datasets:
			self.items.update(dataset.speakers())

		self.items = list(self.items)
		self.items.sort()
		self.dico = {}

		# Assign ordered indicies to each speaker indicies
		for i, item in enumerate(self.items):
			self.dico[int(item)] = i

	def __iter__(self):
		return self

	def get_labels(self):
		return self.dico

from data_tools import read_metadata, males_keys, females_keys
if __name__ == "__main__":

	###
	### TEST
	##
	# H5_dic = read_metadata()
	# print H5_dic
	# chunk_size = 512*100

	# males = H5PY_RW('test_raw.h5py', subset = males_keys(H5_dic))
	# fem = H5PY_RW('test_raw.h5py', subset = females_keys(H5_dic))

	# print 'Data with', len(H5_dic), 'male and female speakers'
	# print males.length(), 'elements'
	# print fem.length(), 'elements'

	# mixed_data = Mixer([males, fem], chunk_size= chunk_size, with_mask=False, with_inputs=True, shuffling=True)

	# batch_size = 128

	# mixed_data.adjust_split_size_to_batchsize(batch_size)
	# nb_batches = mixed_data.nb_batches(batch_size)

	# nb_to_speaker = mixed_data.dico
	# id_f = []
	# id_m = []

	# for i in range(nb_batches):
	# 		_,_, ind = mixed_data.get_batch(batch_size)
	# 		for key, id_ in nb_to_speaker.iteritems():
	# 			for b in range(batch_size):
	# 				if id_ == ind[b][0]:
	# 					if b == 0:
	# 						id_m += [key]
	# 					assert H5_dic[str(key)]['sex'] == 'M' 
	# 				if id_ == ind[b][1]:
	# 					if b == 0:
	# 						id_f += [key]
	# 					assert H5_dic[str(key)]['sex'] == 'F' 

	# mixed_data.select_split(1)
	# x_non_test , x_test , _ = mixed_data.get_only_first_items(8)
	# mixed_data.select_split(0)

	# for j in range(2):
	# 	for i in range(nb_batches):
	# 		_,_, ind = mixed_data.get_batch(batch_size)
	# 		for key, id_ in nb_to_speaker.iteritems():
	# 			if id_ == ind[0][0]:
	# 				assert id_m[i] == key 
	# 			if id_ == ind[0][1]:
	# 				assert id_f[i] == key
	H5PY_RW.create_raw_audio_dataset('train-clean-100-8-s.h5', subset='train-clean-100')
	# print nb_to_speaker.values()
	# print ind

	# H5_dic = read_metadata()
	# males = H5PY_RW('dev-clean.h5', subset = males_keys(H5_dic))
