# -*- coding: utf-8 -*-
import h5py
import numpy as np
import data_tools
import soundfile as sf
import os
import config
from utils.audio import create_spectrogram
from utils.tools import print_progress
from tqdm import tqdm 
import copy

class ConsistentRandom:
	def __init__(self, seed):
		self.previous_state = np.random.get_state()
		np.random.seed(seed)
		self.state = np.random.get_state()

	def __enter__(self):
		np.random.set_state(self.state)

	def __exit__(self, exc_type, exc_value, traceback):
		np.random.set_state(self.previous_state)


class Dataset(object):

	def __init__(self, ratio=[0.90, 0.05, 0.05], sex=['M', 'F'], **kwargs):
		"""
		Inputs:
			path: path to the h5 file
			subset: subset of speakers used, default = None (all in the file)
		"""
		print kwargs
		np.random.seed(config.seed)
		self.nb_speakers = kwargs['nb_speakers']
		self.sex = sex
		self.batch_size = kwargs['batch_size']
		self.chunk_size = kwargs['chunk_size']

		self.TRAIN = 0
		self.VALID = 1
		self.TEST = 2


		# TODO 
		metadata = data_tools.read_metadata()

		if sex != ['M', 'F'] and sex != ['F', 'M'] and sex != ['M'] and sex != ['F']:
			raise Exception('Sex must be ["M","F"] |  ["F","M"] | ["M"] | [F"]')

		self.key_to_index = {}
		j = 0
		if 'M' in sex:
			self.M = data_tools.males_keys(metadata)
			for k in self.M:
				self.key_to_index[k] = j
				j += 1 
		if 'F' in sex:
			self.F = data_tools.females_keys(metadata)
			for k in self.F:
				self.key_to_index[k] = j
				j += 1

		self.tot_speakers = j

		self.file = h5py.File(kwargs['dataset'], 'r')


		# Define all the items related to each key/speaker
		self.total_items = []

		if 'M' in sex:
			for key in self.M:
				for val in self.file[key]:
					chunks = self.file['/'.join([key,val])].shape[0]//self.chunk_size
					self.total_items += ['/'.join([key,val,str(i)]) for i in range(chunks)]

		if 'F' in sex:
			for key in self.F:
				for val in self.file[key]:
					chunks = self.file['/'.join([key,val])].shape[0]//self.chunk_size
					self.total_items += ['/'.join([key,val,str(i)]) for i in range(chunks)]

		np.random.shuffle(self.total_items)

		L = len(self.total_items)
		# Shuffle all the items

		# Training / Valid / Test Separation
		train = self.create_tree(self.total_items[:int(L*ratio[0])])
		valid = self.create_tree(self.total_items[int(L*ratio[0]):int(L*(ratio[0]+ratio[1]))])
		test = self.create_tree(self.total_items[int(L*(ratio[0]+ratio[1])):])
		self.items = [train, valid, test]
		# Init Seed here

	def __iter__(self):
		self.used = copy.deepcopy(self.items[self.index])
		return self

	def create_tree(self, items_list):

		items = {'M':{}, 'F':{}}
		tot_M = 0
		tot_F = 0
		for item in tqdm(items_list, desc='Creating Dataset'):
			splits = item.split('/')
			key = splits[0]
			if 'M' in self.sex and key in self.M:
				if key in items['M']:
					items['M'][key].append(item)
				else:
					items['M'][key] = [item]
				tot_M += 1
			if 'F' in self.sex and key in self.F:
				if key in items['F']:
					items['F'][key].append(item)
				else:
					items['F'][key] = [item]
				tot_F += 1

		if len(self.sex) > 1:
			if tot_M < tot_F:
				D = tot_F - tot_M
				K = items['F'].keys()
				L = len(K)
				for i in range(D):
					l = items['F'][K[i%L]]
					l.remove(np.random.choice(l))
					if len(l) == 0:
						del items['F'][K[i%L]]
					tot_F -= 1
			else:
				D = tot_M - tot_F
				K = items['M'].keys()
				L = len(K)
				for i in range(D):
					l = items['M'][K[i%L]]
					l.remove(np.random.choice(l))
					if len(l) == 0:
						del items['M'][K[i%L]]
					tot_M -= 1
		print items
		items['tot'] = tot_F + tot_M
		return items

	def get_batch(self, index, batch_size, fake=False):
		with ConsistentRandom(config.seed):
			used = copy.deepcopy(self.items[index])
			while True:
				batch = ([], [], [])
				for i in range(batch_size):
					if fake: 
						self.next_item(used, fake)
					else: 
						mix, non_mix, I = self.next_item(used, fake)
						batch[0].append(mix)
						batch[1].append(non_mix)
						batch[2].append(I)
				yield batch

	def next_item(self, used, fake=False):

		genre = np.random.choice(self.sex, self.nb_speakers)

		mix = []
		for s in self.sex:
			nb = sum(map(int,genre == s))
			if nb > len(used[s].keys()):
				raise StopIteration()

			keys = np.random.choice(used[s].keys(), nb, replace=False)

			for key in keys:
				choice = np.random.choice(used[s][key])	
				mix.append(choice)
				used[s][key].remove(choice)
				if len(used[s][key]) == 0:
					del used[s][key]

		if not fake:
			# TODO MIXING TYPE !
			mix_array = np.zeros((self.chunk_size))
			# non_mix_array = np.zeros((self.nb_speakers, self.chunk_size))
			# indices = np.zeros((self.nb_speakers), dtype=int)
			# mix_array = np.zeros()
			non_mix_array = []
			indices = []
			for i, m in enumerate(mix):
				splits = m.split('/')
				key_index = self.key_to_index[splits[0]]
				chunk = int(splits[-1])
				item_path = '/'.join(splits[:-1])

				item = self.file[item_path][chunk*self.chunk_size:(chunk+1)*self.chunk_size]
				mix_array +=item

				non_mix_array.append(item)
				indices.append(key_index)

			return mix_array, non_mix_array, indices

	def nb_batch(self, batch_size):
		i = 0 
		for _ in tqdm(self.get_batch(self.TRAIN, batch_size, fake=True), desc='Counting batches'):
			i+=1
		return i

	def empty_next(self):
		genre = np.random.choice(self.sex, self.nb_speakers)

		mix = []
		for s in self.sex:
			nb = sum(map(int,genre == s))
			if nb > len(self.used[s].keys()):
				self.used = copy.deepcopy(self.items)
				raise StopIteration()

			keys = np.random.choice(self.used[s].keys(), nb, replace=False)

			for key in keys:
				choice = np.random.choice(self.used[s][key])	
				mix.append(choice)
				self.used[s][key].remove(choice)
				if len(self.used[s][key]) == 0:
					del self.used[s][key]
		return

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


if __name__ == "__main__":
	###
	### TEST
	##
	d = Dataset(dataset="h5py_files/train-clean-100-8-s.h5", chunk_size=20480, batch_size=100, nb_speakers=2)
	print 'NB BATCHE', d.nb_batch(batch_size=1)
	for i ,(x_mix, x_non_mix, I) in enumerate(d.get_batch(d.TRAIN, 1)):
		print I
		if i%10 == 0:
			for i ,(x_mix, x_non_mix, I) in enumerate(d.get_batch(d.VALID, 1)):
				print I
	for i ,(x_mix, x_non_mix, I) in enumerate(d.get_batch(d.TRAIN, 1)):
		print I
