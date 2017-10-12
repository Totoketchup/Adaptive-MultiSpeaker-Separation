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

class H5PY_RW:
	def __init__(self):
		self.current = None


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
		speakers_info = data_tools.read_data_header(subset)

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

	def create_raw_audio_dataset(self, output_fn, subset=config.data_subset, data_root=config.data_root):
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
		speakers_info = data_tools.read_data_header(subset)

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
								raw_audio, sr = sf.read(path)

								# raw_audio = downsample(raw_audio, sr, config.fs)

								raw_audio = (raw_audio - np.mean(raw_audio))/np.std(raw_audio)

								data_file[key].create_dataset(file,
									data=raw_audio,
									compression="gzip",
									compression_opts=0)

					print_progress(i + 1, len(elements['chapters']), prefix = 'Speaker '+key+' :', suffix = 'Complete')


		print 'Dataset for the subset: ' + subset + ' has been built'

	def open_h5_dataset(self, filename, subset=None):
		"""
		Open a LibriSpeech H5PY file.
		
		Inputs:
			filename: name of h5 file
			subset: subset of speakers used, default = None (all in the file)
		"""
		path = os.path.join(config.workdir, filename)
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

		self.raw_items = items
		self.items = items
		self.index_item = 0
		return self


	def set_chunk(self, chunk_size):
		"""
		Define the size of the chunk: each data is chunked with 'chunk_size'
		"""

		self.chunk_size = chunk_size
		items = []

		# Chunk along the first axis (might add a parameter for this , TODO)
		for item in self.raw_items:
			L = self.h5[item].shape[0]//chunk_size
			items += [item +'/' + str(part) for part in range(L)]

		# Update the items into chunked items
		self.items = items
		return self

	def next(self):
		"""
		Return next chunked item
		"""
		item_path = self.items[self.index_item]
		split = item_path.split('/')

		if hasattr(self, 'chunk_size'):
			# Which part to chunk
			i = int(split[2])

			# Cut the data according to the chunk
			item_path = '/'.join(split[:2])
			X = self.h5[item_path][i*self.chunk_size : (i+1)*self.chunk_size]
		else:
		 	X = self.h5[item_path]

		# Speaker indice
		key = int(split[0])

		self.index_item+=1
		if self.index_item >= len(self.items):
			self.index_item = 0

		return X, key

	def next_in_split(self, splits, split_index):
		"""
		Return next chunked item in the indicated split
		Input:
			splits: ratio of each split (array)
			index: split index
		"""
		if not hasattr(self, 'index_item_split'):
			self.index_item_split = np.zeros((len(splits),), dtype = np.int32)
			for i in range(1,len(splits)):
				self.index_item_split[i] = int(sum(splits[0:i])*len(self.items))


		item_path = self.items[self.index_item_split[split_index]]
		split = item_path.split('/')

		# Speaker indice
		key = int(split[0])

		if hasattr(self, 'chunk_size'):
			# Which part to chunk
			i = int(split[2])

			# Cut the data according to the chunk
			item_path = '/'.join(split[:2])
			X = self.h5[item_path][i*self.chunk_size : (i+1)*self.chunk_size]
		else:
		 	X = self.h5[item_path]

		self.index_item_split[split_index]+=1
		if self.index_item_split[split_index] >= int(sum(splits[0:split_index+1])*len(self.items)):
			self.index_item_split[split_index] = int(sum(splits[0:split_index])*len(self.items))

		return X, key


	def shuffle(self):
		np.random.shuffle(self.items)
		return self

	def speakers(self):
		return self.keys

	def length(self):
		return len(self.items)

	def __iter__(self):
		return self




class Mixer:

	def __init__(self, datasets, with_mask=True, with_inputs=False, splits = [0.8, 0.1, 0.1], mixing_type='add', mask_positive_value=1, mask_negative_value=-1):
		"""
		Mix multiple H5PY file reader
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

	def shuffle(self):
		for dataset in self.datasets:
			dataset.shuffle()

	def select_split(self, index):
		self.split_index = index

	def next(self):
		X_d = []
		key_d = []

		for dataset in self.datasets:
			X, key = dataset.next_in_split(self.splits, self.split_index)
			X_d.append(X)
			key_d.append(key)

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
			return X_mix, Y, key_d

		if self.with_inputs:
			return X_non_mix, X_mix, key_d
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

		return np.array(X), np.array(Ind)

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

if __name__ == "__main__":
	H5 = H5PY_RW()
	H5.create_raw_audio_dataset('test_raw_16k.h5py')

# # for (X, key) in H5:
# #     continue
# H5.create_h5_dataset('test2.h5py')