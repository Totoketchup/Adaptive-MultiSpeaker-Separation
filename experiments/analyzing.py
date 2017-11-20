import tensorflow as tf 
import os 
import config
from utils.postprocessing.representation import *
# from models import Adapt

## Here we plot the T-SNE of some important Tensors like the speaker centroids or the output
## of embeddings from batches.

####
#### MODEL CONFIG
####

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
full_id = "AdaptiveNet-lucky-brook-0190" + config_id
model_meta_path = os.path.join(config.log_dir, config_model["type"], full_id, 'model.ckpt.meta')
model_path = os.path.join(config.log_dir, config_model["type"], full_id, 'model.ckpt')

sess = tf.Session()

importer = tf.train.import_meta_graph(model_meta_path)
importer.restore(sess, model_path)

centroids = [v for v in tf.global_variables() if v.name == "speaker_centroids:0"][0]

with sess.as_default():
	centroids = centroids.eval()

from data.dataset import *
from data.data_tools import read_metadata

H5_dic = read_metadata()
chunk_size = 512*40
males = H5PY_RW('test_raw.h5py', subset = males_keys(H5_dic))
fem = H5PY_RW('test_raw.h5py', subset = females_keys(H5_dic))

print 'Data with', len(H5_dic), 'male and female speakers'
print males.length(), 'elements'
print fem.length(), 'elements'

mixed_data = Mixer([males, fem], chunk_size= chunk_size, with_mask=False, with_inputs=True, shuffling=True)

batch_size = 128
speaker_to_index = mixed_data.dico
id_f = []
id_m = []
tmp = {v: k for k, v in speaker_to_index.iteritems()}
index_to_speaker = np.zeros(40, np.int32)
for i in range(40):
	index_to_speaker[i] = tmp[i]

labels = [H5_dic[str(int(v))]['sex'] for v in index_to_speaker]

plot_TSNE(centroids, 2, labels)
plot_PCA(centroids, 2, labels)

