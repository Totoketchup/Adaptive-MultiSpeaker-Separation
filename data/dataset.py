import h5py
import numpy as np
import data_tools
import soundfile as sf
import os
import config
from utils.audio import create_spectrogram

class H5PY_RW:
    def __init__(self):
        pass

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
                                    data=spec.astype(np.complex64),
                                    compression="gzip",
                                    dtype=np.complex64,
                                    compression_opts=0)
            

        print 'Dataset for the subset: '+subset+' has been built'

    def open_h5_dataset(self, filename):
        self.h5 = h5py.File(filename,'r')
        self.keys = [key for key in self.h5]
        items = []
        for key in self.h5:
            items += [key + '/' + val  for val in self.h5[key]]
        self.items = items


H5 = H5PY_RW()
H5.open_h5_dataset('test.h5py')
print H5.items
