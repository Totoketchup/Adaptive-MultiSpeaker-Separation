import matplotlib
matplotlib.use('Agg')
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


config_model = {}
config_model["type"] = "L41_train_front"
config_model["batch_size"] = 64
config_model["chunk_size"] = 512*40
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
full_id = "AdaptiveNet-muddy-mountain-6335" + config_id
path = os.path.join(config.model_root, 'log', config_model["type"], full_id)

l = os.listdir(path)
centroids_filename = [ v for v in l if 'centroids' in v ]
centroids_filename = sorted(centroids_filename, key=lambda name: int(name[10:-4]))

total = len(centroids_filename)

to_plot = []

for filename in tqdm(centroids_filename[0:total], desc='Reading TSNE centroids'):
	to_plot += [PCA_representation(np.load(os.path.join(path, filename)),2)]
to_plot = np.array(to_plot)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


fig, ax = plt.subplots()
scat = ax.scatter([], [], s=10)
plt.xlim(-3., 3.)
plt.ylim(-3., 3.)
plt.xlabel('X1')
plt.ylabel('X2')
plt.style.use('ggplot')
plt.legend(handles=[red_patch, blue_patch])

def update_plot(i):
	print i
	scat.set_offsets(to_plot[i])
	scat.set_color(colors)
	plt.title('PCA of speakers centroids on 2 components at step = {}'.format(i))
	return scat,

ani = animation.FuncAnimation(fig, update_plot, blit=False, interval=100, frames=xrange(total))

ani.save(os.path.join('/output', 'gif', 'centroids','anim.gif'), writer="imagemagick")



# print 'Creating plots'
# for i in tqdm(range(8)):
# 	centroids = np.load(centroids_path+'-{}.npy'.format(i))

# 	tsne = TSNE_representation(centroids, 2)

# 	tsne_x = tsne[:,0]
# 	tsne_y = tsne[:,1]
	
# 	plt.scatter(tsne_x, tsne_y, color=colors)
# 	plt.legend(handles=[red_patch, blue_patch])
# 	plt.title('TSNE of speakers centroids on 2 components at step = {}'.format(i))
# 	plt.axis([-1.5,1.5,-1.5,1.5])
# 	plt.xlabel('X1')
# 	plt.ylabel('X2')

# 	plt.savefig(os.path.join(config.workdir, 'gif', 'centroids','plot{}.png'.format(i)))
# 	plt.close()

# print 'Creating GIF'
# os.system('convert -loop 0 -delay 10 gif/centroids/*.png gif/centroids/centroids.gif')

# print 'Delete images'
# for i in tqdm(range(8)):
# 	os.remove(os.path.join(config.workdir, 'gif', 'centroids','plot{}.png'.format(i)))
