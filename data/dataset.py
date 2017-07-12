import h5py
import numpy as np
import data_tools
import soundfile as sf
import os
import config
from utils.audio import create_spectrogram

class H5PY_RW:
    def __init__(self):
        self.current = None

    def create_h5_dataset(self, output_fn, subset=config.data_subset, data_root=config.data_root):
        speakers_info = data_tools.read_data_header(subset)
        with h5py.File(output_fn,'w') as data_file:
            for (key, elements) in speakers_info.items():
                if key not in data_file:
                    data_file.create_group(key)
                print 'Speaker '+key
                folder = data_root+'/'+subset+'/'+key # speaker folder
                for chapter in elements['chapters']: # for all chapters read by this speaker
                    print '-- Chapter '+chapter
                    for root, dirs, files in os.walk(folder+'/'+chapter): # find all .flac audio
                        for file in files:
                            if file.endswith(".flac"):
                                print '------ Track '+file
                                path = os.path.join(root,file)
                                raw_audio, samplerate = sf.read(path)

                                _, _, spec = create_spectrogram(raw_audio, samplerate)

                                data_file[key].create_dataset(file,
                                    data=spec.T.astype(np.complex64),
                                    compression="gzip",
                                    dtype=np.complex64,
                                    compression_opts=0)
            

        print 'Dataset for the subset: ' + subset + ' has been built'

    def open_h5_dataset(self, filename):
        self.h5 = h5py.File(filename, 'r')
        self.keys = [key for key in self.h5]
        items = []
        for key in self.h5:
            items += [key + '/' + val  for val in self.h5[key]]
        self.raw_items = items
        self.index_key = 0
        self.index_item = 0

    def set_chunk(self, chunk_size):
        self.chunk_size = chunk_size
        items = []
        for item in self.raw_items:
            L = self.h5[item].shape[0]//chunk_size
            items += [item +'/' + str(part) for part in range(L)]
        self.items = items

    def next(self):
        item_path = self.items[self.index_item]
        split = item_path.split('/')
        key = split[0]
        i = int(split[2])
        item_path = '/'.join(split[:2])
        X = self.h5[item_path][i*self.chunk_size : (i+1)*self.chunk_size]

        self.index_item+=1
        if self.index_item >= len(self.items):
            self.index_item = 0

        return X, key

    def shuffle(self):
        permutation = np.random.shuffle(self.items)


    def __iter__(self):
        return self




class Mixer:
    def __init__(datasets):
        pass


    def next():
        pass

# H5 = H5PY_RW()
# # H5.open_h5_dataset('test.h5py')

# # for (X, key) in H5:
# #     continue
# H5.create_h5_dataset('test.h5py')