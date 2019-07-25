import numpy as np

import keras.backend as K
from keras.layers import Input, Layer
from keras.models import Model
from keras.initializers import Constant, RandomNormal
from keras.regularizers import l2

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.python.training import optimizer
from tensorflow.train import AdamOptimizer

import itertools

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

	uu = K.sum(x ** 2 / \
		source_sigma, 
		axis=-1, keepdims=True) # assume sigma inv is diagonal

	log_det = K.sum(K.log(target_sigma), axis=-1, keepdims=True) - \
		K.sum(K.log(source_sigma), axis=-1, keepdims=True)
	# log_det = K.log(K.prod(target_sigma, axis=-1, keepdims=True) / \
	# 	K.prod(source_sigma, axis=-1, keepdims=True))

	return 0.5 * (trace + uu - k - log_det) 

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape) 
	return K.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def parallel_transport(p, q, x):
	alpha = -minkowski_dot(p, q)
	return x + minkowski_dot(q - alpha * p, x) * (p + q)  / \
		K.maximum(alpha + 1, K.epsilon()) 
		# (alpha + 1)

def normalise_to_hyperboloid(x):
	return x / K.maximum( K.sqrt( K.maximum( -minkowski_dot(x, x), 0.) ), K.epsilon())

def exponential_mapping( p, x ):

	norm_x = K.sqrt( K.maximum( 
		minkowski_dot(x, x), 0. ) ) 
	####################################################
	# exp_map_p = tf.cosh(norm_x) * p

	# idx = tf.cast( tf.where(norm_x > K.cast(0., K.floatx()), )[:,0], tf.int64)
	# non_zero_norm = tf.gather(norm_x, idx)
	# z = tf.gather(x, idx) / non_zero_norm

	# updates = tf.sinh(non_zero_norm) * z
	# dense_shape = tf.cast( tf.shape(p), tf.int64)
	# exp_map_x = tf.scatter_nd(indices=idx[:,None], updates=updates, shape=dense_shape)
	
	# exp_map = exp_map_p + exp_map_x 
	#####################################################
	z = x / K.maximum(norm_x, K.epsilon()) # unit norm 
	# r = K.minimum(norm_x, K.ones_like(norm_x)*.001)
	r = norm_x
	exp_map = tf.cosh(r) * p + tf.sinh(r) * z
	#####################################################
	exp_map = normalise_to_hyperboloid(exp_map) # account for floating point imprecision

	return exp_map

def logarithmic_map(p, x):
	assert len(p.shape) == len(x.shape)

	alpha = -minkowski_dot(p, x)

	alpha = 1 + K.maximum(alpha - 1, K.epsilon())

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

class HyperboloidEmbeddingLayer(Layer):
	
	def __init__(self, 
		num_nodes, 
		embedding_dim, 
		**kwargs):
		super(HyperboloidEmbeddingLayer, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='embedding', 
		  shape=(self.num_nodes, self.embedding_dim),
		  initializer=hyperboloid_initializer,
		  trainable=True)
		super(HyperboloidEmbeddingLayer, self).build(input_shape)

	def call(self, idx):

		embedding = tf.gather(self.embedding, idx)

		return embedding

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], self.embedding_dim + 1)
	
	def get_config(self):
		base_config = super(HyperboloidEmbeddingLayer, self).get_config()
		base_config.update({"num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim})
		return base_config

class HyperboloidEmbeddingLayerGaussian(Layer):
	
	def __init__(self, 
		num_nodes, 
		embedding_dim, 
		**kwargs):
		super(HyperboloidEmbeddingLayerGaussian, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim
		self.mu_zero = K.constant(np.append(np.zeros((1, 1, self.embedding_dim)), np.ones((1,1,1)), axis=-1))

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='hyperbolic_embedding', 
		  shape=(self.num_nodes, self.embedding_dim),
		  initializer=hyperboloid_initializer,
		  trainable=True)
		assert self.embedding.shape[1] == self.embedding_dim + 1
		self.covariance = self.add_weight(name='euclidean_covariance', 
		  shape=(self.num_nodes, self.embedding_dim),
		  initializer="glorot_normal",
		#   initializer="ones",
		#   initializer=RandomNormal(mean=min_, 
		#   stddev=1e-30,), 
		#   regularizer=l2(1e-3),
		  trainable=True)
		super(HyperboloidEmbeddingLayerGaussian, self).build(input_shape)

	def call(self, idx):

		source_embedding = tf.gather(self.embedding, idx[:,:1])
		target_embedding = tf.gather(self.embedding, idx[:,1:])

		to_tangent_space = logarithmic_map(source_embedding, 
			target_embedding)
		to_tangent_space_mu_zero = parallel_transport(source_embedding, 
			self.mu_zero, 
			to_tangent_space)

		sigmas = tf.gather(self.covariance, idx)  

		sigmas = K.elu(sigmas, alpha=1-K.epsilon()) + 1

		kds = kullback_leibler_divergence(\
			to_tangent_space_mu_zero, 
			sigmas=sigmas)

		kds = K.squeeze(kds, axis=-1)

		return kds 

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1] - 1, )
	
	def get_config(self):
		base_config = super(HyperboloidEmbeddingLayerGaussian, 
			self).get_config()
		base_config.update({"num_nodes": self.num_nodes, 
			"embedding_dim": self.embedding_dim})
		return base_config

def asym_hyperbolic_loss(y_true, y_pred):
	# d = K.int_shape(y_pred)[1]
	d = 3
	pred = K.concatenate([tf.nn.sparse_softmax_cross_entropy_with_logits(\
		labels=y_true[:,0], logits=-K.concatenate([y_pred[:,i:i+1], y_pred[:,j:j+1]], axis=-1, ))[:,None]
		for i, j in itertools.combinations(range(d), 2)]
		+ [tf.nn.sparse_softmax_cross_entropy_with_logits(\
		labels=y_true[:,0], logits=-K.concatenate([y_pred[:,i:i+1], y_pred[:,j:j+1]], axis=-1, ))[:,None]
		for i, j in itertools.product(range(d), range(d, d+10))]
		, axis=-1)
	return K.mean(K.mean(pred, axis=-1))
	# pred = K.concatenate( [y_pred[:,i:i+1]**2 + K.exp(-y_pred[:,j:j+1]) 
		# for i in range(d) for j in range(i+1, d)], axis=-1)
	# return K.mean( K.mean( pred,	axis=-1, keepdims=True ) )
	# return K.mean(y_pred[:,:1] ** 2 + \
	# 	K.mean(K.exp(-y_pred[:,1:]), axis=-1, keepdims=True))
	# return -K.mean(K.exp(y_pred[:,:1]) - K.mean(K.exp(y_pred[:,1:]), axis=-1, keepdims=True))
	# return K.mean(1 - 1 / (1 - y_pred[:,:1]) + K.mean(1 / (1 -y_pred[:,1:]), axis=-1, keepdims=True))
	# return K.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
	# 	labels=y_true[:,0], logits=-y_pred))

class ExponentialMappingOptimizer(optimizer.Optimizer):
	
	def __init__(self, 
		lr=0.1, 
		use_locking=False,
		name="ExponentialMappingOptimizer"):
		super(ExponentialMappingOptimizer, self).__init__(use_locking, name)
		self.lr = lr
		self.adam = AdamOptimizer()

	def _apply_dense(self, grad, var):
		assert False
		spacial_grad = grad[:,:-1]
		t_grad = -1 * grad[:,-1:]
		
		ambient_grad = tf.concat([spacial_grad, t_grad], 
			axis=-1)
		tangent_grad = self.project_onto_tangent_space(var, 
			ambient_grad)
		
		exp_map = exponential_mapping(var, 
			- self.lr * tangent_grad)
		
		return tf.assign(var, exp_map)
		
	def _apply_sparse(self, grad, var):


		if "hyperbolic" in var.name:

			indices = grad.indices
			values = grad.values

			p = tf.gather(var, indices, 
				name="gather_apply_sparse")

			spacial_grad = values[:, :-1]
			t_grad = -1 * values[:, -1:]

			ambient_grad = tf.concat([spacial_grad, t_grad], 
				axis=-1, name="optimizer_concat")

			tangent_grad = self.project_onto_tangent_space(p, 
				ambient_grad)
			exp_map = exponential_mapping(p, 
				- self.lr * tangent_grad)

			return tf.scatter_update(ref=var, 
				indices=indices, updates=exp_map, 
				name="scatter_update")

		else:
			# euclidean update using Adam optimizer
			return self.adam.apply_gradients( [(grad, var), ] )

	def project_onto_tangent_space(self, 
		hyperboloid_point, 
		minkowski_ambient):
		return minkowski_ambient + minkowski_dot(hyperboloid_point, minkowski_ambient) * hyperboloid_point
   
def build_hyperboloid_asym_model(num_nodes, 
	embedding_dim, 
	num_negative_samples, 
	lr=1e-1, ):
	x = Input((1 + 3 + num_negative_samples, ),
		dtype=tf.int64)
	hyperboloid_embedding = HyperboloidEmbeddingLayerGaussian(num_nodes, embedding_dim)(x)

	model = Model(x, hyperboloid_embedding)

	optimizer = ExponentialMappingOptimizer(lr=lr,)

	model.compile(optimizer=optimizer, 
		loss=asym_hyperbolic_loss,
		target_tensors=[ tf.placeholder(dtype=tf.int64), ])

	return model
