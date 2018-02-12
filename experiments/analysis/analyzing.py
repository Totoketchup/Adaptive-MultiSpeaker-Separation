import tensorflow as tf 
import os 
import config
from utils.postprocessing.representation import *
from data.dataset import *
from data.data_tools import read_metadata
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm
# from models import Adapt

## Here we plot the T-SNE of some important Tensors like the speaker centroids or the output
## of embeddings from batches.

H5_dic = read_metadata()
chunk_size = 512*40
males = H5PY_RW('test_raw.h5py', subset = males_keys(H5_dic))
fem = H5PY_RW('test_raw.h5py', subset = females_keys(H5_dic))
mixed_data = Mixer([males, fem], chunk_size= chunk_size, with_mask=False, with_inputs=True, shuffling=True)

speaker_to_index = mixed_data.dico
tmp = {v: k for k, v in speaker_to_index.iteritems()}
index_to_speaker = np.zeros(40, np.int32)
for i in range(40):
	index_to_speaker[i] = tmp[i]
labels = [H5_dic[str(int(v))]['sex'] for v in index_to_speaker]

colors = ['r' if v =='M' else 'b' for v in labels]
red_patch = mpatches.Patch(color='red', label='Men')
blue_patch = mpatches.Patch(color='blue', label='Women')

####
#### MODEL CONFIG
# ####

# config_model = {}
# config_model["type"] = "L41_train_front"
# config_model["batch_size"] = 64
# config_model["chunk_size"] = 512*10
# config_model["N"] = 256
# config_model["maxpool"] = 256
# config_model["window"] = 1024
# config_model["smooth_size"] = 10
# config_model["alpha"] = 0.01
# config_model["reg"] = 1e-3
# config_model["beta"] = 0.05
# config_model["rho"] = 0.01
# config_model["same_filter"] = True
# config_model["optimizer"] = 'Adam'

# config_id = ''.join('-{}={}-'.format(key, val) for key, val in sorted(config_model.items()))
# full_id = "AdaptiveNet-frosty-fire-4612" + config_id
# model_meta_path = os.path.join(config.log_dir, config_model["type"], full_id, 'model.ckpt.meta')
# model_path = os.path.join(config.log_dir, config_model["type"], full_id, 'model.ckpt')


config_model = {}
config_model["type"] = "L41_train_front"
config_model["batch_size"] = 64
config_model["chunk_size"] = 512*10
config_model["N"] = 256
config_model["maxpool"] = 256
config_model["window"] = 1024
config_model["smooth_size"] = 10
config_model["alpha"] = 0.01
config_model["reg"] = 1e-3
config_model["beta"] = 0.05
config_model["rho"] = 0.01
config_model["same_filter"] = True
config_model["optimizer"] = 'Adam'

config_id = ''.join('-{}={}-'.format(key, val) for key, val in sorted(config_model.items()))
full_id = "AdaptiveNet-soft-lab-2491" + config_id
centroids_path = os.path.join(config.log_dir, config_model["type"], full_id, 'centroids')



print 'Creating plots'
for i in tqdm(range(1519)):
	centroids = np.load(centroids_path+'-{}.npy'.format(i))

	tsne = TSNE_representation(centroids, 2)

	tsne_x = tsne[:,0]
	tsne_y = tsne[:,1]
	
	plt.scatter(tsne_x, tsne_y, color=colors)
	plt.legend(handles=[red_patch, blue_patch])
	plt.title('TSNE of speakers centroids on 2 components at step = {}'.format(i))
	plt.axis([-1.5,1.5,-1.5,1.5])
	plt.xlabel('X1')
	plt.ylabel('X2')

	plt.savefig(os.path.join(config.workdir, 'gif', 'centroids','plot{}.png'.format(i)))
	plt.close()

print 'Creating GIF'
os.system('convert -loop 0 -delay 10 gif/centroids/*.png gif/centroids/centroids.gif')

print 'Delete images'
for i in tqdm(range(1519)):
	os.remove(os.path.join(config.workdir, 'gif', 'centroids','plot{}.png'.format(i)))
