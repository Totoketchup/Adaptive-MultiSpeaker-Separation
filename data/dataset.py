# -*- coding: utf-8 -*-
import h5py
import numpy as np
import data_tools
import os
import config
from tqdm import tqdm 
import copy
import time
from librosa.core import resample,load
import tensorflow as tf

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
		self.total_items = self.total_items

		L = len(self.total_items)
		# Shuffle all the items

		# Training / Valid / Test Separation
		train = self.create_tree(self.total_items[:int(L*ratio[0])])
		valid = self.create_tree(self.total_items[int(L*ratio[0]):int(L*(ratio[0]+ratio[1]))])
		test = self.create_tree(self.total_items[int(L*(ratio[0]+ratio[1])):])
		
		self.train = TreeIterator(train, self)
		self.valid = TreeIterator(valid, self)
		self.test = TreeIterator(test, self)

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

class TreeIterator(object):

	def __init__(self, tree, dataset):
		self.tree = tree
		self.sex = dataset.sex
		self.nb_speakers = dataset.nb_speakers
		self.chunk_size = dataset.chunk_size
		self.key_to_index = dataset.key_to_index
		self.file = dataset.file
		self.no_random_picking = dataset.no_random_picking

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
			mix_array = np.zeros((self.chunk_size))
			non_mix_array = [[] for _ in range(len(mix))]
			indices = [[] for _ in range(len(mix))]

			# Mixing all the items
			for i, m in enumerate(mix):
				splits = m.split('/')
				key_index = self.key_to_index[splits[0]]
				chunk = int(splits[-1])
				item_path = '/'.join(splits[:-1])

				item = self.file[item_path][chunk*self.chunk_size:(chunk+1)*self.chunk_size]
				mix_array += item

				non_mix_array[i] = item 
				indices[i] = key_index

			return mix_array, non_mix_array, indices


	def __call__(self):
		used = copy.deepcopy(self.tree)
		while True:
			try:
				yield self.next_item(used)
			except Exception:
				raise
				used = copy.deepcopy(self.tree)
			else:
				pass
			finally:
				pass



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


def create_mix(output_fn, chunk_size, batch_size, nb_speakers, sex, no_random_picking):
	# Extract the information about this subset (speakers, chapters)
	# Dictionary with the following shape: 
	# {speaker_key: {chapters: [...], sex:'M/F', ... } }
	data = Dataset(dataset="h5py_files/train-clean-100-8-s.h5", 
		chunk_size=chunk_size, 
		batch_size=batch_size, 
		nb_speakers=nb_speakers,
		sex=sex,
		no_random_picking=no_random_picking)

	with h5py.File(output_fn,'w') as data_file:

		f = [("train", data.TRAIN),("test",data.TEST), ("valid",data.VALID)]

		for group_name, data_split in f:
			print group_name
			data_file.create_group(group_name)
			train_mix = data_file[group_name].create_dataset("mix",
							shape=(batch_size, chunk_size),
							maxshape=(None, chunk_size),
							compression="gzip",
							chunks=(64, chunk_size),
							dtype='float32')
			train_non_mix = data_file[group_name].create_dataset("non_mix",
							shape=(batch_size, nb_speakers, chunk_size),
							maxshape=(None, nb_speakers, chunk_size),
							compression="gzip",
							chunks=(64, nb_speakers, chunk_size),
							dtype='float32')
			train_index = data_file[group_name].create_dataset("ind",
							shape=(batch_size, nb_speakers),
							maxshape=(None, nb_speakers),
							compression="gzip",
							chunks=(64, nb_speakers),
							dtype='int32')
			size = batch_size
			for i ,(mix, non_mix, index) in enumerate(data.get_batch(data_split, batch_size)):
				train_mix[i*batch_size:(i+1)*batch_size] = mix
				train_non_mix[i*batch_size:(i+1)*batch_size] = non_mix
				train_index[i*batch_size:(i+1)*batch_size] = index
				
				size = size + batch_size
				train_mix.resize((size, chunk_size))
				train_non_mix.resize((size, nb_speakers, chunk_size))
				train_index.resize((size, nb_speakers))

def create_tfrecord_file(output_fn, chunk_size, batch_size, nb_speakers, sex, no_random_picking):
	# Extract the information about this subset (speakers, chapters)
	# Dictionary with the following shape: 
	# {speaker_key: {chapters: [...], sex:'M/F', ... } }
	data = Dataset(dataset="h5py_files/train-clean-100-8-s.h5", 
		chunk_size=chunk_size, 
		batch_size=batch_size, 
		nb_speakers=nb_speakers,
		sex=sex,
		no_random_picking=no_random_picking)

	f = [("train", data.TRAIN),("test",data.TEST), ("valid",data.VALID)]

	for group_name, data_split in f:
		
		writer = tf.python_io.TFRecordWriter(group_name +'.tfrecords')
		print group_name
		for i ,(mix, non_mix, index) in enumerate(data.get_batch(data_split, batch_size)):
			mix_raw = mix[0].astype(np.float32).tostring()
			non_mix_raw = non_mix[0].astype(np.float32).tostring()
			index = np.array(index[0]).tostring()

			feature = tf.train.Example(features=tf.train.Features(
							feature = { 'chunk_size':tf.train.Feature(int64_list=tf.train.Int64List(value=[chunk_size])),
										'nb_speakers':tf.train.Feature(int64_list=tf.train.Int64List(value=[nb_speakers])),
										'mix':tf.train.Feature(bytes_list=tf.train.BytesList(value=[mix_raw])),
										'non_mix':tf.train.Feature(bytes_list=tf.train.BytesList(value=[non_mix_raw])),
										'ind':tf.train.Feature(bytes_list=tf.train.BytesList(value=[index]))
									}))

			writer.write(feature.SerializeToString())

		writer.close()

def create_tfrecord_file_2(output_fn, chunk_size, batch_size, nb_speakers, sex, no_random_picking):
	# Extract the information about this subset (speakers, chapters)
	# Dictionary with the following shape: 
	# {speaker_key: {chapters: [...], sex:'M/F', ... } }
	data = Dataset(dataset="h5py_files/train-clean-100-8-s.h5", 
		chunk_size=chunk_size, 
		batch_size=batch_size, 
		nb_speakers=nb_speakers,
		sex=sex,
		no_random_picking=no_random_picking)

	f = [("train", data.TRAIN),("test", data.TEST), ("valid", data.VALID)]

	for group_name, data_split in f:
		
		writer = tf.python_io.TFRecordWriter(group_name +'.tfrecords')
		print group_name
		for i ,(_, non_mix, index) in enumerate(data.get_batch(data_split, batch_size)):
			non_mix_raw = non_mix[0].astype(np.float32).tostring()
			index = np.array(index[0]).tostring()

			feature = tf.train.Example(features=tf.train.Features(
							feature = { 'chunk_size':tf.train.Feature(int64_list=tf.train.Int64List(value=[chunk_size])),
										'nb_speakers':tf.train.Feature(int64_list=tf.train.Int64List(value=[nb_speakers])),
										'non_mix':tf.train.Feature(bytes_list=tf.train.BytesList(value=[non_mix_raw])),
										'ind':tf.train.Feature(bytes_list=tf.train.BytesList(value=[index]))
									}))

			writer.write(feature.SerializeToString())

		writer.close()


def from_flac_to_tfrecords(train_r=0.8, valid_test_r=0.2):
	# Extract the information about this subset (speakers, chapters)
	# Dictionary with the following shape: 
	# {speaker_key: {chapters: [...], sex:'M/F', ... } }
	folder = config.data_root+'/'+config.data_subset
	speakers_info = data_tools.read_metadata(config.data_subset)
	keys_to_index = {}
	for i, key in enumerate(speakers_info.keys()):
		keys_to_index[key] = i

	sex = ['M' for i in range(len(speakers_info))]
	for k, v in speakers_info.items():
		i = keys_to_index[k]
		sex[i] = v['sex']

	np.save('genders_index.arr', sex)
	# exit()

	allfiles = np.array([os.path.join(r,f) for r,dirs,files in os.walk(folder) for f in files if f.endswith(".flac")])
	L = len(allfiles)
	np.random.shuffle(allfiles)
	train = allfiles[:int(L*train_r)]
	valid = allfiles[int(L*train_r):int(L*(train_r+valid_test_r/2))]
	test = allfiles[int(L*(train_r+valid_test_r/2)):]
	
	print len(train), len(valid), len(test)

	for group_name, data_split in [("train", train),("test", test), ("valid", valid)]:

		for s in ['M', 'F']:

			writer = tf.python_io.TFRecordWriter(group_name + '_' + s +'.tfrecords')

			for file in data_split:

				splits = file.split('/')
				key = splits[-3]
				sex = speakers_info[key]['sex']

				if sex == s:

					raw_audio, sr = load(file, sr=16000)
					raw_audio = resample(raw_audio, sr, config.fs)
					raw_audio = raw_audio.astype(np.float32).tostring()

					feature = tf.train.Example(features=tf.train.Features(
						feature = { 'audio' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_audio])),
									'key' : tf.train.Feature(int64_list=tf.train.Int64List(value=[keys_to_index[key]]))
					}))
					print group_name, s, key, keys_to_index[key]
					writer.write(feature.SerializeToString())

			writer.close()

def decode(serialized_example):
	features = tf.parse_single_example(
		serialized_example,
		features={
			'key': tf.FixedLenFeature([], tf.int64),
			'audio':tf.FixedLenFeature([], tf.string),
		})

	audio = tf.decode_raw(features['audio'], tf.float32)

	return audio, features['key']

def normalize(non_mix, ind):
	mean, var = tf.nn.moments(non_mix, -1, keep_dims=True)
	non_mix = (non_mix - mean)/tf.sqrt(var)

	return non_mix, ind

def mix(*non_mix):
	non_mix, keys = zip(*non_mix)
	non_mix = tf.stack(non_mix)
	keys = tf.stack(keys)
	mix = tf.reduce_sum(non_mix, 0)
	print mix, non_mix, keys
	return mix, non_mix, keys


def is_long_enough(audio, key, chunk_size):
	return tf.less(chunk_size, tf.shape(audio)[0])

def filtering(mix, non_mix, keys):
	L = tf.shape(keys)[0]
	tiled = tf.tile(tf.expand_dims(keys, 1), [1, L])
	sums = tf.reduce_sum(tf.cast(tf.equal(keys, tiled), tf.int32))
	sums = tf.reduce_sum(sums)
	return tf.equal(L, sums)

def chunk(audio, key, chunk_size):
	L = tf.shape(audio)[0]
	nb = tf.floordiv(L, chunk_size)
	chunks = tf.map_fn(lambda i : audio[i*chunk_size:(i+1)*chunk_size], tf.range(nb), dtype=tf.float32)
	return chunks, tf.tile(tf.expand_dims(key, 0), [nb]) 	# [N, chunk_size]

def setshape(mix, non_mix, keys, chunk, N):
	mix = tf.reshape(mix, [chunk])
	non_mix = tf.reshape(non_mix, [N, chunk])
	keys = tf.reshape(keys, [N])
	return mix, non_mix, keys

class TFDataset(object):

	def get_data(self, name, seed):
		return tf.data.TFRecordDataset(name) \
				.map(decode) \
				.shuffle(1000, seed=seed) \
				.filter(lambda a, k : is_long_enough(a, k, self.chunk_size)) \
				.map(lambda a, k : chunk(a, k, self.chunk_size)) \
				.apply(tf.contrib.data.unbatch()) \
				.shuffle(1000, seed=seed)


	def __init__(self, **kwargs):

		batch_size = kwargs['batch_size']
		self.normalize = kwargs['dataset_normalize']
		self.chunk_size = kwargs['chunk_size']
		N = kwargs['nb_speakers']

		print kwargs['sex']
		with tf.name_scope('dataset'):

			# MALES
			if 'M' in kwargs['sex']:
				train_M = lambda i : self.get_data('train_M.tfrecords', i)
				valid_M = lambda i : self.get_data('valid_M.tfrecords', i) 
				test_M = lambda i : self.get_data('test_M.tfrecords', i)

			# FEMALES
			if 'F' in kwargs['sex']:
				train_F = lambda i : self.get_data('train_F.tfrecords', i)
				valid_F = lambda i : self.get_data('valid_F.tfrecords', i)
				test_F = lambda i : self.get_data('test_F.tfrecords', i)

			# MIXING
			if 'M' in kwargs['sex'] and 'F' in kwargs['sex']:
				train_list = tuple([train_M(i) if i%2 == 0 else train_F(i) for i in range(N)])
				valid_list = tuple([valid_M(i) if i%2 == 0 else valid_F(i) for i in range(N)])
				test_list = tuple([test_M(i) if i%2 == 0 else test_F(i) for i in range(N)])
			elif 'M' in kwargs['sex']:
				train_list = tuple([train_M(i) for i in range(N)])
				valid_list = tuple([valid_M(i) for i in range(N)])
				test_list = tuple([test_M(i) for i in range(N)])
			else:
				train_list = tuple([train_F(i) for i in range(N)])
				valid_list = tuple([valid_F(i) for i in range(N)])
				test_list = tuple([test_F(i) for i in range(N)])
			
			train_mix = tf.data.TFRecordDataset.zip(train_list)
			valid_mix = tf.data.TFRecordDataset.zip(valid_list)
			test_mix = tf.data.TFRecordDataset.zip(test_list)
			
			train_mix = train_mix.map(mix)
			train_mix = train_mix.filter(filtering)
			train_mix = train_mix.batch(batch_size)
			train_mix = train_mix.prefetch(1)

			valid_mix = valid_mix.map(mix)
			valid_mix = valid_mix.filter(filtering)
			valid_mix = valid_mix.batch(batch_size)
			valid_mix = valid_mix.prefetch(1)

			test_mix = test_mix.map(mix)
			test_mix = test_mix.filter(filtering)
			test_mix = test_mix.batch(batch_size)
			test_mix = test_mix.prefetch(1)

			self.handle = tf.placeholder(tf.string, shape=[])
			iterator = tf.data.Iterator.from_string_handle(
				self.handle, train_mix.output_types, train_mix.output_shapes)
			self.next_element = iterator.get_next()

			self.training_iterator = train_mix.make_initializable_iterator()
			self.validation_iterator = valid_mix.make_initializable_iterator()
			self.test_iterator = test_mix.make_initializable_iterator()

			self.training_initializer = self.training_iterator.initializer
			self.validation_initializer = self.validation_iterator.initializer
			self.test_initializer = self.test_iterator.initializer

			self.next_mix, self.next_non_mix, self.next_ind = self.next_element

	def init_handle(self):
		sess = tf.get_default_session()
		self.training_handle = sess.run(self.training_iterator.string_handle())
		self.validation_handle = sess.run(self.validation_iterator.string_handle())
		self.test_handle = sess.run(self.test_iterator.string_handle())

	def get_handle(self, split):
		if split == 'train':
			return self.training_handle
		elif split == 'valid':
			return self.validation_handle
		elif split == 'test':
			return self.test_handle

	def get_initializer(self, split):
		if split == 'train':
			return self.training_initializer
		elif split == 'valid':
			return self.validation_initializer
		elif split == 'test':
			return self.test_initializer

	def length(self, split):
		count = 0
		sess = tf.get_default_session()
		sess.run(self.get_initializer(split))
		try:
			while True:
				sess.run(self.next_element, feed_dict={self.handle: self.get_handle(split)})
				count += 1
		except Exception:
			return count

	def initialize(self, sess, split):
		sess.run(self.initializer,feed_dict={self.split: split})

class MixGenerator(object):

	def __init__(self, **kwargs):
		batch_size = kwargs['batch_size']
		self.normalize = kwargs['dataset_normalize']
		chunk_size = kwargs['chunk_size']
		N = kwargs['nb_speakers']

		d = Dataset(**kwargs)
		
		train_mix = tf.data.Dataset.from_generator(d.train, (tf.float32, tf.float32, tf.int64))
		valid_mix = tf.data.Dataset.from_generator(d.valid, (tf.float32, tf.float32, tf.int64))
		test_mix = tf.data.Dataset.from_generator(d.test, (tf.float32, tf.float32, tf.int64))

		train_mix = train_mix.map(lambda x, y, z: setshape(x,y,z,chunk_size, N))
		train_mix = train_mix.batch(batch_size)
		train_mix = train_mix.prefetch(1)	
		print '---------------------------'
		print train_mix
		
		valid_mix = valid_mix.map(lambda x, y, z: setshape(x,y,z,chunk_size, N))
		valid_mix = valid_mix.batch(batch_size)
		valid_mix = valid_mix.prefetch(1)

		test_mix = test_mix.map(lambda x, y, z: setshape(x,y,z,chunk_size, N))
		test_mix = test_mix.batch(batch_size)
		test_mix = test_mix.prefetch(1)

		self.handle = tf.placeholder(tf.string, shape=[])
		iterator = tf.data.Iterator.from_string_handle(
			self.handle, train_mix.output_types, train_mix.output_shapes)
		self.next_element = iterator.get_next()

		self.training_iterator = train_mix.make_initializable_iterator()
		self.validation_iterator = valid_mix.make_initializable_iterator()
		self.test_iterator = test_mix.make_initializable_iterator()

		self.training_initializer = self.training_iterator.initializer
		self.validation_initializer = self.validation_iterator.initializer
		self.test_initializer = self.test_iterator.initializer

		self.next_mix, self.next_non_mix, self.next_ind = self.next_element

	def init_handle(self):
		sess = tf.get_default_session()
		self.training_handle = sess.run(self.training_iterator.string_handle())
		self.validation_handle = sess.run(self.validation_iterator.string_handle())
		self.test_handle = sess.run(self.test_iterator.string_handle())

	def get_handle(self, split):
		if split == 'train':
			return self.training_handle
		elif split == 'valid':
			return self.validation_handle
		elif split == 'test':
			return self.test_handle

	def get_initializer(self, split):
		if split == 'train':
			return self.training_initializer
		elif split == 'valid':
			return self.validation_initializer
		elif split == 'test':
			return self.test_initializer

	def length(self, split):
		count = 0
		sess = tf.get_default_session()
		sess.run(self.get_initializer(split))
		try:
			while True:
				sess.run(self.next_element, feed_dict={self.handle: self.get_handle(split)})
				count += 1
		except Exception:
			return count

	def initialize(self, sess, split):
		sess.run(self.initializer,feed_dict={self.split: split})

if __name__ == "__main__":
	###
	### TEST
	##

	from_flac_to_tfrecords()

	# ds = TFDataset(dataset ='h5py_files/train-clean-100-8-s.h5', batch_size=256, dataset_normalize=False, nb_speakers=2, sex=['M', 'F'], chunk_size=20480, no_random_picking=True)
	# with tf.Session().as_default() as sess:
	# 	ds.init_handle() 
	# 	# L = ds.length('train')
	# 	# print ds.length('train'), ds.length('test'), ds.length('valid')
	# 	sess.run(ds.training_initializer)
	# 	for i in range(10):
	# 		t = time.time()
	# 		value = sess.run(ds.next_element, feed_dict={ds.handle: ds.get_handle('train')})
	# 		print value[2]
	# 		# print time.time() - t

	# 	ds.init_handle() 
	# 	sess.run(ds.training_initializer)
	# 	for i in range(10):
	# 		t = time.time()
	# 		value = sess.run(ds.next_element, feed_dict={ds.handle: ds.get_handle('train')})
	# 		print time.time() - t

	# 		sess.run(ds.validation_initializer)
	# 		for _ in range(10):
	# 			t = time.time()
	# 			value = sess.run(ds.next_element, feed_dict={ds.handle: ds.get_handle('valid')})
	# 			print '--- ', time.time() - t
			

