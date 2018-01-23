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

bins_filename = [ v for v in l if 'bins' in v ]
bins_filename = sorted(bins_filename, key=lambda name: int(name[5:-4]))
total = len(bins_filename)

labels_filename = [ v for v in l if 'labels' in v ]
labels_filename = sorted(labels_filename, key=lambda name: int(name[7:-4]))

red_patch = mpatches.Patch(color='red', label='Men')
blue_patch = mpatches.Patch(color='blue', label='Women')

to_plot = []
colors = []


print 'TOTAL:', total

for i, filename in tqdm(enumerate(bins_filename[:total]), total=total, desc='Reading bins embeddings'):
	data = np.load(os.path.join(path, filename))
	reduced = range(data.shape[1])

	labels = np.load(os.path.join(path, labels_filename[i]))

	data = np.reshape(data[0], [-1, 40])
	to_plot += [PCA_representation(data,3)]

	colors += [np.reshape(labels[0], (-1))]
to_plot = np.array(to_plot)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# total = 10

# to_plot = [np.random.rand(5120, 3) for i in range(total)]
# colors = [['r' if i%2==0 else 'b' for i in range(5120)]for i in range(total)]

# fig, ax = plt.subplots()
# scat = ax.scatter([], [], s=10)
fig = matplotlib.pyplot.figure()
ax3d = Axes3D(fig)
scat3D = ax3d.scatter([],[],[], s=1)
ttl = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes)

def update_plot(i):
	print i, to_plot[i].shape
	ttl.set_text('PCA on 3 components at step = {}'.format(i*20))
	scat3D._offsets3d = np.transpose(to_plot[i])

def init():
	scat3D.set_offsets([[],[], []])
	ax3d.set_xlim(-1.,2.)
	ax3d.set_ylim(-0.5,0.7)
	ax3d.set_zlim(-1.,0.75)
	plt.legend(handles=[red_patch, blue_patch])

ani = animation.FuncAnimation(fig, update_plot, init_func=init, blit=False, interval=100, frames=xrange(total))

ani.save(os.path.join('/output', 'gif', 'bins','anim2.gif'), writer="imagemagick")



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
