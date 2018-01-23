import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

def ICA_representation(data, n_components):
	ica = FastICA(n_components=n_components, random_state=42)
	return ica.fit_transform(data)

def PCA_representation(data, n_components):
	pca = PCA(n_components=n_components, random_state=42)
	return pca.fit_transform(data)

def TSNE_representation(data, n_components):
	model = TSNE(n_components=n_components, n_iter=50000, learning_rate=10.0, random_state=42)
	return model.fit_transform(data) 

def plot_PCA(data, n_components, labels, name='PCA Representation'):
	pca = PCA_representation(data, n_components)
	pca_x = pca[:,0]
	pca_y = pca[:,1]
	colors = ['r' if v =='M' else 'b' for v in labels]
	plt.scatter(pca_x, pca_y, color=colors)
	# plt.colorbar(ticks=range(10))
	plt.show()

def plot_ICA(data, n_components, labels, name='ICA Representation'):
	pca = ICA_representation(data, n_components)
	pca_x = pca[:,0]
	pca_y = pca[:,1]
	colors = ['r' if v =='M' else 'b' for v in labels]
	plt.scatter(pca_x, pca_y, color=colors)
	plt.show()

def plot_TSNE(data, n_components, labels, name='TSNE Representation'):
	tsne = TSNE_representation(data, n_components)
	tsne_x = tsne[:,0]
	tsne_y = tsne[:,1]
	colors = ['r' if v =='M' else 'b' for v in labels]
	plt.scatter(tsne_x, tsne_y, color=colors)
	# plt.colorbar(ticks=range(10))
	plt.show()
