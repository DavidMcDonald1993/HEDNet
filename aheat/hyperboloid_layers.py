import numpy as np

import keras.backend as K
from keras.layers import Input, Layer, Concatenate
from keras.models import Model
from keras.initializers import Constant, RandomNormal
from keras.regularizers import l2

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.python.training import optimizer
# from tensorflow.keras.optimizers import SGD

# import itertools

# min_ = 1e-0

# def euclidean_pdf(x, mu, sigma):

# 	# ignore zero t coordinate
# 	x = x[...,:-1]

# 	k = K.int_shape(x)[-1]

# 	x_minus_mu = x - mu

# 	uu = K.sum(x_minus_mu ** 2 / K.maximum(sigma, K.epsilon()), 
# 		axis=-1, keepdims=True) # assume sigma inv is diagonal
# 	uu = K.exp(-0.5 * uu)
# 	dd = K.sqrt((2 * np.pi) ** k * K.prod(sigma, axis=-1, keepdims=True))

# 	return uu / dd

# def log_likelihood(x, mu, sigma):

# 	# ignore zero t coordinate
# 	x = x[...,:-1]

# 	k = K.int_shape(x)[-1]

# 	x_minus_mu = x - mu

# 	sigma = K.maximum(sigma, K.epsilon())

# 	uu = K.sum(x_minus_mu ** 2 / sigma, 
# 		axis=-1, keepdims=True) # assume sigma inv is diagonal

# 	log_sigma_det = K.sum(K.log(sigma), axis=-1, keepdims=True)

# 	return - 0.5 * (log_sigma_det + uu + k * K.log(K.constant(2. * np.pi)))

def kullback_leibler_divergence(x, sigmas):

	# ignore zero t coordinate
	x = x[...,:-1]

	k = K.int_shape(x)[-1]

	sigmas = K.maximum(sigmas, K.epsilon())

	source_sigma = sigmas[:,:1]
	target_sigma = sigmas[:,1:]

	trace = K.sum(target_sigma / \
		source_sigma, 
		axis=-1, keepdims=True)

	mu_sq_diff = K.sum(x ** 2 / \
		source_sigma, 
		axis=-1, keepdims=True) # assume sigma inv is diagonal

	log_det = K.sum(K.log(target_sigma), axis=-1, keepdims=True) - \
		K.sum(K.log(source_sigma), axis=-1, keepdims=True)
	# log_det = K.log(K.prod(target_sigma, axis=-1, keepdims=True) / \
	# 	K.prod(source_sigma, axis=-1, keepdims=True))
	# log_det = K.sum(K.log(target_sigma / \
		# source_sigma), axis=-1, keepdims=True)

	return 0.5 * (trace + mu_sq_diff - k - log_det) 

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape) 
	return K.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def parallel_transport(p, q, x):
	alpha = -minkowski_dot(p, q)
	return x + minkowski_dot(q - alpha * p, x) * (p + q)  / \
		K.maximum(alpha + 1, K.epsilon()) 

def logarithmic_map(p, x):
	assert len(p.shape) == len(x.shape)

	alpha = -minkowski_dot(p, x)

	alpha = K.maximum(alpha, 1+K.epsilon())

	return tf.acosh(alpha) * (x - alpha * p) / \
		K.maximum(K.sqrt(K.maximum(alpha ** 2 - 1., 0.)), K.epsilon())		  
		# K.sqrt(alpha ** 2 - 1)		  

def hyperboloid_initializer(shape, r_max=1e-2):

	def poincare_ball_to_hyperboloid(X, append_t=True):
		x = 2 * X
		t = 1. + K.sum(K.square(X), axis=-1, keepdims=True)
		if append_t:
			x = K.concatenate([x, t], axis=-1)
		return 1 / (1. - K.sum(K.square(X), axis=-1, keepdims=True)) * x

	# def sphere_uniform_sample(shape, r_max):
	# 	num_samples, dim = shape
	# 	X = tf.random_normal(shape=shape, dtype=K.floatx())
	# 	X_norm = K.sqrt(K.sum(K.square(X), axis=-1, keepdims=True))
	# 	U = tf.random_uniform(shape=(num_samples, 1), dtype=K.floatx())
	# 	return r_max * U ** (1./dim) * X / X_norm

	# w = sphere_uniform_sample(shape, r_max=r_max)
	w = tf.random_uniform(shape=shape, minval=-r_max, maxval=r_max, dtype=K.floatx())
	return poincare_ball_to_hyperboloid(w)

# class HyperboloidEmbeddingLayer(Layer):
	
# 	def __init__(self, 
# 		num_nodes, 
# 		embedding_dim, 
# 		**kwargs):
# 		super(HyperboloidEmbeddingLayer, self).__init__(**kwargs)
# 		self.num_nodes = num_nodes
# 		self.embedding_dim = embedding_dim

# 	def build(self, input_shape):
# 		# Create a trainable weight variable for this layer.
# 		self.embedding = self.add_weight(name='embedding', 
# 		  shape=(self.num_nodes, self.embedding_dim),
# 		  initializer=hyperboloid_initializer,
# 		  trainable=True)
# 		super(HyperboloidEmbeddingLayer, self).build(input_shape)

# 	def call(self, idx):

# 		embedding = tf.gather(self.embedding, idx)

# 		return embedding

# 	def compute_output_shape(self, input_shape):
# 		return (input_shape[0], input_shape[1], self.embedding_dim + 1)
	
# 	def get_config(self):
# 		base_config = super(HyperboloidEmbeddingLayer, self).get_config()
# 		base_config.update({"num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim})
# 		return base_config

class HyperboloidGaussianEmbeddingLayer(Layer):
	
	def __init__(self, 
		num_nodes, 
		embedding_dim, 
		**kwargs):
		super(HyperboloidGaussianEmbeddingLayer, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim
		self.mu_zero = K.constant(np.append(np.zeros((1, 1, self.embedding_dim)), np.ones((1,1,1)), axis=-1))

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='hyperbolic_embedding', 
		  shape=(self.num_nodes, self.embedding_dim),
		  initializer=hyperboloid_initializer,
		  trainable=False)
		assert self.embedding.shape[1] == self.embedding_dim + 1
		self.covariance = self.add_weight(name='euclidean_covariance', 
		  shape=(self.num_nodes, self.embedding_dim),
		#   initializer="glorot_uniform",
			initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64),
		#   initializer="ones",
		#   initializer=RandomNormal(mean=min_, 
		#   stddev=1e-30,), 
		  trainable=True)
		super(HyperboloidGaussianEmbeddingLayer, self).build(input_shape)

	def call(self, idx):

		source_embedding = tf.gather(self.embedding, idx[:,:1])
		target_embedding = tf.gather(self.embedding, idx[:,1:])

		to_tangent_space = logarithmic_map(source_embedding, 
			target_embedding)
		to_tangent_space_mu_zero = parallel_transport(source_embedding, 
			self.mu_zero, 
			to_tangent_space)

		sigmas = tf.gather(self.covariance, idx)  

		sigmas = K.elu(sigmas, alpha=.9) + 1

		kds = kullback_leibler_divergence(\
			to_tangent_space_mu_zero, 
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




   

