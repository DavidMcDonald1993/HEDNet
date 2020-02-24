import numpy as np

import keras.backend as K
from keras.layers import Input, Layer, Concatenate
from keras.models import Model
from keras.initializers import Constant, RandomNormal
from keras.regularizers import l2

import tensorflow as tf

import functools

def kullback_leibler_divergence(mus, sigmas):

	# ignore zero t coordinate
	mus = mus[...,:-1]

	# source_mus, target_mus = mus[:,:1], mus[:,1:]

	k = K.int_shape(mus)[-1]

	sigmas = K.maximum(sigmas, K.epsilon())

	source_sigma = sigmas[:,:1]
	target_sigma = sigmas[:,1:]

	sigma_ratio = target_sigma / source_sigma
	sigma_ratio = K.maximum(sigma_ratio, K.epsilon())

	trace_fac = K.sum(sigma_ratio,
		axis=-1, keepdims=True)

	# mu_sq_diff = K.sum(K.square(target_mus - source_mus) / \
	# 	source_sigma,
	# 	axis=-1, keepdims=True) # assume sigma is diagonal
	mu_sq_diff = K.sum(K.square(mus) / \
		source_sigma,
		axis=-1, keepdims=True) # assume sigma is diagonal

	log_det = K.sum(K.log(sigma_ratio),
		axis=-1, keepdims=True)

	return 0.5 * (trace_fac + mu_sq_diff - k - log_det)

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape)
	return K.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def minkowski_norm(x):
	return K.sqrt(K.maximum(
		minkowski_dot(x, x), 0.))

def parallel_transport(p, q, x):
	alpha = -minkowski_dot(p, q)
	alpha = K.maximum(alpha, 1+K.epsilon())

	return x + minkowski_dot(q - alpha * p, x) * (p + q)  / \
		K.maximum(alpha + 1, K.epsilon())

def logarithmic_map(p, x):
	assert len(p.shape) == len(x.shape)

	alpha = -minkowski_dot(p, x)# + K.epsilon()

	alpha = K.maximum(alpha, 1 + K.epsilon())

	ret = tf.acosh(alpha) * (x - alpha * p) / \
		K.maximum(K.sqrt(K.maximum(alpha ** 2 - 1., 0.)),
		K.epsilon())

	return ret

def hyperboloid_initializer(shape, r_max=5e-3):

	def poincare_ball_to_hyperboloid(X, append_t=True):
		x = 2 * X
		t = 1. + K.sum(K.square(X), axis=-1, keepdims=True)
		if append_t:
			x = K.concatenate([x, t], axis=-1)
		return 1 / (1. - K.sum(K.square(X), axis=-1, keepdims=True)) * x

	w = tf.random_uniform(shape=shape, 
		minval=-r_max, 
		maxval=r_max, 
		dtype=K.floatx())
	return poincare_ball_to_hyperboloid(w)

class HyperboloidGaussianEmbeddingLayer(Layer):

	def __init__(self,
		num_nodes,
		embedding_dim,
		**kwargs):
		super(HyperboloidGaussianEmbeddingLayer, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim
		self.mu_zero = K.constant(\
			np.append(np.zeros((1, 1, self.embedding_dim)), 
				np.ones((1,1,1)), axis=-1))

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='hyperbolic_embedding',
			shape=(self.num_nodes, self.embedding_dim),
			initializer=hyperboloid_initializer,
			trainable=True)
		assert self.embedding.shape[1] == self.embedding_dim + 1
		self.sigmas = self.add_weight(name='euclidean_covariance',
			shape=(self.num_nodes, self.embedding_dim),
			initializer=functools.partial(
				tf.random_normal, 
				stddev=1e-3,
				dtype=K.floatx()),
			regularizer=l2(1e-4),
			trainable=True)
		super(HyperboloidGaussianEmbeddingLayer, self).build(input_shape)

	def call(self, idx):

		source_embedding = tf.gather(self.embedding, 
			idx[:,:1])
		target_embedding = tf.gather(self.embedding, 
			idx[:,1:])

		to_tangent_space = logarithmic_map(
			source_embedding,
			target_embedding)

		to_tangent_space_mu_zero = parallel_transport(
			source_embedding,
			self.mu_zero,
			to_tangent_space)

		sigmas = tf.gather(self.sigmas, idx)

		sigmas = K.elu(sigmas) + 1.

		kds = kullback_leibler_divergence(
			mus=to_tangent_space_mu_zero,
			sigmas=sigmas)

		kds = K.squeeze(kds, axis=-1)

		return kds

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1] - 1, )

	def get_config(self):
		base_config = super(HyperboloidGaussianEmbeddingLayer,
			self).get_config()
		base_config.update({"num_nodes": self.num_nodes,
			"embedding_dim": self.embedding_dim})
		return base_config