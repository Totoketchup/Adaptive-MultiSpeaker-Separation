import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def PCA_representation(data, n_components):
	pca = PCA(n_components=n_components)
	return pca.fit_transform(data)

def TSNE_representation(data, n_components):
	model = TSNE(n_components=n_components, random_state=0)
	return model.fit_transform(data) 

def plot_PCA(data, n_components, name='PCA Representation'):
	pca = PCA_representation(data, n_components)

def plot_TSNE(data, n_components, name='TSNE Representation'):
	tsne = TSNE_representation(data, n_components)
