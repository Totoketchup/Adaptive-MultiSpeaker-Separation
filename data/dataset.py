# -*- coding: utf-8 -*-
import h5py
import numpy as np
import data_tools
import os
import config
from tqdm import tqdm 
import copy

"""
Class used to have a consistent Randomness between Training/Validation/Test for
different batch size
"""
class ConsistentRandom:
	def __init__(self, seed):
		# Create a new Random State and save the current one
		self.previous_state = np.random.get_state()
		np.random.seed(seed)
		self.state = np.random.get_state()

	def __enter__(self):
		# Apply the new state
		np.random.set_state(self.state)

	def __exit__(self, exc_type, exc_value, traceback):
		# Restore the previous state
		np.random.set_state(self.previous_state)


class Dataset(object):

	def __init__(self, ratio=[0.90, 0.05, 0.05], **kwargs):
		"""
		Inputs:
			ratio: ratio for train / valid / test set
			kwargs: Dataset parameters
		"""

		np.random.seed(config.seed)

		self.nb_speakers = kwargs['nb_speakers']
		self.sex = kwargs['sex']
		self.batch_size = kwargs['batch_size']
		self.chunk_size = kwargs['chunk_size']
		self.no_random_picking = kwargs['no_random_picking']

		# Flags for Training/Validation/Testing sets
		self.TRAIN = 0
		self.VALID = 1
		self.TEST = 2

		# TODO 
		metadata = data_tools.read_metadata()

		if self.sex != ['M', 'F'] and self.sex != ['F', 'M'] and self.sex != ['M'] and self.sex != ['F']:
			raise Exception('Sex must be ["M","F"] |  ["F","M"] | ["M"] | [F"]')

		# Create a key to speaker index dictionnary
		# And count the numbers of speakers
		self.key_to_index = {}
		self.sex_to_keys = {}
		j = 0

		if 'M' in self.sex:
			M = data_tools.males_keys(metadata)
			self.sex_to_keys['M'] = M
			for k in M:
				self.key_to_index[k] = j
				j += 1 
		if 'F' in self.sex:
			F = data_tools.females_keys(metadata)
			self.sex_to_keys['F'] = F
			for k in F:
				self.key_to_index[k] = j
				j += 1

		self.tot_speakers = j

		self.file = h5py.File(kwargs['dataset'], 'r')


		# Define all the items related to each key/speaker
		self.total_items = []

		for key in self.key_to_index.keys():
			for val in self.file[key]:
				# Get one file related to a speaker and check how many chunks can be obtained
				# with the current chunk size
				chunks = self.file['/'.join([key,val])].shape[0]//self.chunk_size
				# Add each possible chunks in the items with the following form:
				# 'key/file/#chunk'
				self.total_items += ['/'.join([key,val,str(i)]) for i in range(chunks)]

		np.random.shuffle(self.total_items)
		self.total_items = self.total_items[0:1000] 	

		L = len(self.total_items)
		# Shuffle all the items

		# Training / Valid / Test Separation
		train = self.create_tree(self.total_items[:int(L*ratio[0])])
		valid = self.create_tree(self.total_items[int(L*ratio[0]):int(L*(ratio[0]+ratio[1]))])
		test = self.create_tree(self.total_items[int(L*(ratio[0]+ratio[1])):])
		self.items = [train, valid, test]

	def __iter__(self):
		self.used = copy.deepcopy(self.items[self.index])
		return self

	def create_tree(self, items_list):

		items = {'M':{}, 'F':{}}
		tot = {'M':0, 'F':0}

		# Putting Men items in 'M' dictionnary and Female items in the 'F' one
		for item in tqdm(items_list, desc='Creating Dataset'):
			splits = item.split('/')
			key = splits[0] # Retrieve key
			for s in self.sex:
				if key in self.sex_to_keys[s]:
					if key in items[s]:
						items[s][key].append(item)
					else:
						items[s][key] = [item]
					tot[s] += 1
					break

		# Balancing Women and Men items
		if len(self.sex) > 1:
			if tot['M'] < tot['F']:
				D = tot['F'] - tot['M']
				K = items['F'].keys()
				L = len(K)
				for i in range(D):
					l = items['F'][K[i%L]]
					l.remove(np.random.choice(l))
					if len(l) == 0:
						del items['F'][K[i%L]]
					tot['F'] -= 1
			else:
				D = tot['M'] - tot['F']
				K = items['M'].keys()
				L = len(K)
				for i in range(D):
					l = items['M'][K[i%L]]
					l.remove(np.random.choice(l))
					if len(l) == 0:
						del items['M'][K[i%L]]
					tot['M'] -= 1

		items['tot'] = tot['F'] + tot['M']
		return items

	"""
	Getting a batch from the selected set
	Inputs:
		- index: index of the set self.TRAIN / self.TEST / self.VALID 
		- batch_size
		- fake: True -> Do not return anything (used to count the nb of total batches in an epoch) 
	"""
	def get_batch(self, index, batch_size, fake=False):
		with ConsistentRandom(config.seed):
			used = copy.deepcopy(self.items[index])
			while True:
				mix = []
				non_mix = []
				I = []
				for i in range(batch_size):
					if fake: 
						self.next_item(used, fake)
					else: 
						m, n_m, ind = self.next_item(used, fake)
						mix.append(m)
						non_mix.append(n_m)
						I.append(ind)
				mix = np.array(mix)
				non_mix = np.array(non_mix)
				I = np.array(I)
				yield (mix, non_mix, I)

	def next_item(self, used, fake=False):

		# Random picking or regular picking or the speaker sex
		if not self.no_random_picking or len(self.sex) == 1:
			genre = np.random.choice(self.sex, self.nb_speakers)
		else:
			genre = np.array(['M' if i%2 == 0 else 'F' for i in range(self.nb_speakers)])

		mix = []
		for s in self.sex:
			nb = sum(map(int,genre == s)) # Get the occurence # of 's' in the mix

			# If there is not enough items left, we cannot create new mixtures
			# It's the end of the current epoch
			if nb > len(used[s].keys()):
				raise StopIteration()

			# Select random keys in each sex
			keys = np.random.choice(used[s].keys(), nb, replace=False)

			for key in keys:
				# Select a random chunk and remove it from the list
				choice = np.random.choice(used[s][key])	
				mix.append(choice)
				used[s][key].remove(choice)
				if len(used[s][key]) == 0:
					del used[s][key]

		if not fake:
			# TODO MIXING TYPE !
			mix_array = np.zeros((self.chunk_size))
			non_mix_array = []
			indices = []

			# Mixing all the items
			for i, m in enumerate(mix):
				splits = m.split('/')
				key_index = self.key_to_index[splits[0]]
				chunk = int(splits[-1])
				item_path = '/'.join(splits[:-1])

				item = self.file[item_path][chunk*self.chunk_size:(chunk+1)*self.chunk_size]
				mix_array += item

				non_mix_array.append(item)
				indices.append(key_index)

			return mix_array, non_mix_array, indices

	"""
	Counts the number of batches in the Training Set
	"""
	def nb_batch(self, batch_size):
		i = 0 
		for _ in tqdm(self.get_batch(self.TRAIN, batch_size, fake=True), desc='Counting batches'):
			i+=1
		return i

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


if __name__ == "__main__":
	###
	### TEST
	##
	d = Dataset(dataset="h5py_files/train-clean-100-8-s.h5", chunk_size=20480, batch_size=100, nb_speakers=2)
	print 'NB BATCH', d.nb_batch(batch_size=1)
	for i ,(x_mix, x_non_mix, I) in enumerate(d.get_batch(d.TRAIN, 1)):
		print I
		if i%10 == 0:
			for i ,(x_mix, x_non_mix, I) in enumerate(d.get_batch(d.VALID, 1)):
				print I
	for i ,(x_mix, x_non_mix, I) in enumerate(d.get_batch(d.TRAIN, 1)):
		print I
