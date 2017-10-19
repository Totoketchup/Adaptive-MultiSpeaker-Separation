########################
## DATA CONFIGURATION ##
########################
import os

floydhub = False
output = 'output' if floydhub else ''

workdir = os.path.dirname(__file__)
main_output_dir = os.path.join(workdir, 'output')
if floydhub:
	h5py_root = '/h5py_files'
else:
	h5py_root = os.path.join(workdir, 'h5py_files')
log_dir = os.path.join(main_output_dir,'log')


###
## Raw data
###

data_root = 'data/LibriSpeech'
data_subset = 'dev-clean'
dev_clean_speakers = 40

#########################
## AUDIO CONFIGURATION ##
#########################

fs = 8000 
fftsize = 256
overlap = 2
window = 'hann'


#########################

embedding_size = 40
threshold = 1e-8
chunk_size = 40
batch_size = 32
batch_test = 1
stop_iterations = 10000
max_iterations = 1000000