import tensorflow as tf

from utils.ops import ops
from models.Kmeans_2 import KMeans
import config
from utils.ops.ops import BLSTM, Conv1D, Reshape, Normalize, f_props
class L41Model:
	def __init__(self, input_tensor,
		adapt_front,
		S_tot=config.dev_clean_speakers,
		E=config.embedding_size,
		layer_size=600,
		nonlinearity='logistic',
		normalize=False):
		"""
		Initializes Lab41's clustering model.  Default architecture comes from
		the parameters used by the best performing model in the paper[1].
		[1] Hershey, John., et al. "Deep Clustering: Discriminative embeddings
			for segmentation and separation." Acoustics, Speech, and Signal
			Processing (ICASSP), 2016 IEEE International Conference on. IEEE,
			2016.
		Inputs:
			F: Number of frequency bins in the input data
			num_speakers: Number of unique speakers to train on. only use in
						  training.
			layer_size: Size of BLSTM layers
			embedding_size: Dimension of embedding vector
			nonlinearity: Nonlinearity to use in BLSTM layers (default logistic)
			normalize: Do you normalize vectors coming into the final layer?
					   (default False)
		"""

		self.F = adapt_front.N
		self.num_speakers = S_tot
		self.layer_size = layer_size
		self.embedding_size = E
		self.nonlinearity = nonlinearity
		self.normalize = normalize
		self.B = adapt_front.B
		self.S = adapt_front.S

		self.graph = adapt_front.graph

		with self.graph.as_default():

			self.X, self.X_raw = input_tensor
			print self.X
			with tf.name_scope('create_masks'):
				# # Batch of Masks (bins label)
				# # shape = [ batch size, chunk size, F, S]
				argmax = tf.argmax(self.X_raw, axis=3)
				self.y = tf.one_hot(argmax, 2, 1.0, -1.0)

			# Speakers indicies used in the mixtures
			# shape = [ batch size, #speakers]
			self.I = adapt_front.Ind

			# Define the speaker vectors to use during training
			self.speaker_vectors =tf.Variable(tf.truncated_normal(
								   [self.num_speakers,self.embedding_size],
								   stddev=tf.sqrt(2/float(self.embedding_size))), name='speaker_centroids')

	@ops.scope
	def prediction(self):
		"""
		Construct the op for the network used in [1].  This consists of two
		BLSTM layers followed by a dense layer giving a set of T-F vectors of
		dimension embedding_size
		"""


		shape = tf.shape(self.X)

		layers = [
		BLSTM(self.layer_size, 'BLSTM_1'),
		BLSTM(self.layer_size, 'BLSTM_2'),
		Conv1D([1, self.layer_size, self.embedding_size*self.F]),
		Reshape([shape[0], shape[1], self.F, self.embedding_size]),
		Normalize(3)
		]

		y = f_props(layers, self.X)

		# Normalize the T-F vectors to get the network output
		if self.normalize:
			y = tf.nn.l2_normalize(y, 3)

		return y

	@ops.scope
	def separate(self):
		# TODO for only when Ind is available, speaker info is given
		# Ind [B, S, 1]
		Ind = tf.expand_dims(self.I, 2)

		# U [S_tot, E]
		centroids = tf.gather_nd(self.speaker_vectors, Ind)

		input_kmeans = tf.reshape(self.prediction, [self.B, -1, self.embedding_size])
		kmeans = KMeans(nb_clusters=2, nb_iterations=10, input_tensor=input_kmeans, centroids_init=centroids, latent_space_tensor=self.X)
		_ , labels, soft_labels = kmeans.network
		self.masks = tf.one_hot(labels, 2, 1.0, 0.0)

		separated = tf.reshape(self.X, [self.B, -1, 1])* self.masks # [B ,TF, S] 
		separated = tf.reshape(separated, [self.B, -1, self.F, self.S])
		separated = tf.transpose(separated, [0,3,1,2])
		separated = tf.reshape(separated, [self.B*self.S, -1, self.F, 1])

		return separated

	@ops.scope
	def enhance(self):
		# X_mix = tf.reshape(self.X, [self.B, 1, -1, self.F])
		# X_non_mix = tf.reshape(self.X_raw, [self.B, self.S, -1, self.F])
		# perfect_masks = tf.divide(X_non_mix, X_mix) # [B, S, T ,F] [0 .. 1]

		enhance_input = tf.concat(self.separated, self.X_raw, axis = 3)

	@ops.scope
	def cost(self):
		"""
		Constuct the cost function op for the negative sampling cost
		"""

		# Get the embedded T-F vectors from the network
		embedding = self.prediction

		# Reshape I so that it is of the correct dimension
		I = tf.expand_dims( self.I, axis=2 )

		# Normalize the speaker vectors and collect the speaker vectors
		# correspinding to the speakers in batch
		if self.normalize:
			speaker_vectors = tf.nn.l2_normalize(self.speaker_vectors, 1)
		else:
			speaker_vectors = self.speaker_vectors
		Vspeakers = tf.gather_nd(speaker_vectors, I)

		# Expand the dimensions in preparation for broadcasting
		Vspeakers_broad = tf.expand_dims(Vspeakers, 1)
		Vspeakers_broad = tf.expand_dims(Vspeakers_broad, 1)
		embedding_broad = tf.expand_dims(embedding, 3)

		# Compute the dot product between the emebedding vectors and speaker
		# vectors
		dot = tf.reduce_sum(Vspeakers_broad * embedding_broad, 4)

		# Compute the cost for every element
		cost = -tf.log(tf.nn.sigmoid(self.y * dot))

		# Average the cost over all speakers in the input
		cost = tf.reduce_mean(cost, 3)

		# Average the cost over all batches
		cost = tf.reduce_mean(cost, 0)

		# Average the cost over all T-F elements.  Here is where weighting to
		# account for gradient confidence can occur
		cost = tf.reduce_mean(cost)

		tf.summary.scalar('cost', cost)

		return cost