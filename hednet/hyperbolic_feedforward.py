import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np

from keras.layers import Layer, Input, Lambda
import keras.backend as K
from keras.models import Model
from keras import regularizers

import pandas as pd


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.python.training import optimizer

K.set_floatx("float64")
K.set_epsilon(1e-15)

c = K.constant(1., dtype=K.floatx())

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def poincare_ball_to_hyperboloid(X):
	x = 2 * X
	t = 1. + K.sum(K.square(X), axis=-1, keepdims=True)
	x = K.concatenate([x, t], axis=-1)
	return 1 / (1. - K.sum(K.square(X), axis=-1, keepdims=True)) * x

def minkowski_dot(x, y):
	axes = len(x.shape) - 1, len(y.shape) -1
	return K.batch_dot(x[...,:-1], y[...,:-1], axes=axes) - K.batch_dot(x[...,-1:], y[...,-1:], axes=axes)

def norm( x, axis=-1, keepdims=True):
	return K.sqrt(K.sum(K.square(x), axis=axis, keepdims=keepdims))

def mobius_add( x, y, ):
	xy = K.sum(x * y, axis=-1, keepdims=True)
	norm_x = norm(x, keepdims=True)
	norm_y = norm(y, keepdims=True)
	dd = 1 + 2 * c * xy + c ** 2 * norm_x ** 2 * norm_y ** 2
	uu = (1 + 2 * c * xy + c * norm_y ** 2) * x + (1 - c * norm_x ** 2) * y
	return uu / dd

def mobius_scalar_multiply(r, x, ):
	norm_x = norm(x)
	return 1 / K.sqrt(c) * tf.tanh(r * tf.atanh(K.minimum(K.sqrt(c) * norm_x, 1- 1e-13))) * x / norm_x

def mobius_matrix_vector_multiplication(M, x, ):

	print (M.shape, x.shape)

	Mx = K.batch_dot(x, M, axes=[2, 2])
	norm_x = norm(x)
	norm_Mx = norm(Mx)

	return Mx #/ norm_x

	print (norm_x.shape)
	print (norm_Mx.shape)

	print ((norm_Mx / norm_x).shape)
	# raise SystemExit

	return 1. / K.sqrt(c) * tf.tanh (norm_Mx / norm_x * tf.atanh(K.minimum(K.sqrt(c) * norm_x, 
		1 - 1e-15,))) * Mx / norm_Mx 

def exp_map_x(v, x,):
	norm_v = norm(v)
	return mobius_add(x, K.tanh(K.sqrt(c) * lambda_x(x) * norm_v / 2.) * v / (K.sqrt(c) * norm_v))

def log_map_x( v, x, ):
	minus_x_plus_v = mobius_add(-x, v)
	minus_x_plus_v_norm = norm(minus_x_plus_v)
	return 2 / (K.sqrt(c) * lambda_x(x)) * tf.atanh(K.minimum(K.sqrt(c) * minus_x_plus_v_norm, 
		1- 1e-13)) * minus_x_plus_v / minus_x_plus_v_norm

def exp_map_0( v, ):
	return exp_map_x(v, x=K.zeros_like(v))

def log_map_0( v, ):
	return log_map_x(v, x=K.zeros_like(v))

def mobius_fx( f, x):
	return exp_map_0(f(log_map_0(x)))

def lambda_x(x, axis=-1, ):
	norm_x = norm(x, axis=axis, keepdims=True)
	return 2 / (1 - c * norm_x ** 2)

def parallel_transport(self, v, x):
	return lambda_x(K.zeros_like(x)) / lambda_x(x) * v

class FeedForwardLayer(Layer):

	def __init__(self,
		units,
		c=np.float64(1.),
		activation="sigmoid",
		**kwargs):
		super(FeedForwardLayer, self).__init__(**kwargs)
		self.units = units

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.W = self.add_weight(name='poincare_weight', 
		  shape=( self.units, input_shape[-1]),
		  initializer="uniform",
		  regularizer=regularizers.l2(1e-10),
		  trainable=True) # poincare weight

		self.bias = self.add_weight(name='poincare_bias', 
		  shape=(1, self.units, ),
		  initializer="uniform",
		  trainable=True) # hyperbolic (poincare) weight

		super(FeedForwardLayer, self).build(input_shape)

	def call(self, x):
		if self.activation == "sigmoid":
			f = K.sigmoid
		else:
			raise NotImplementedError
		return self.mobius_fx(f, self.bias_add( self.mobius_matrix_vector_multiplication(self.W, x) , self.bias))


	def compute_output_shape(self, input_shape):
		return tuple(list(input_shape)[:-1] + [self.units])
	
	def get_config(self):
		base_config = super(FeedForwardLayer, self).get_config()
		base_config.update({"units": self.units, "activation": self.activation})
		return base_config


class PoincareEmbeddingLayer(Layer):

	def __init__(self, 
		num_nodes, 
		embedding_dim, 
		**kwargs):
		super(PoincareEmbeddingLayer, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='poincare_embedding', 
		  shape=(self.num_nodes, self.embedding_dim),
		  initializer="uniform",
		  trainable=True)
		super(PoincareEmbeddingLayer, self).build(input_shape)

	def call(self, idx):

		embedding = tf.gather(self.embedding, idx)
		return embedding

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], self.embedding_dim)
	
	def get_config(self):
		base_config = super(PoincareEmbeddingLayer, self).get_config()
		base_config.update({"num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim})
		return base_config


class HyperbolicEmbeddingLayerAsymmetric(Layer):

	def __init__(self, 
		num_nodes, 
		embedding_dim, 
		c=K.constant(np.float64(1.)),
		**kwargs):
		super(HyperbolicEmbeddingLayerAsymmetric, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim
		self.identity = K.constant(np.identity(embedding_dim))

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='poincare_embedding', 
		  shape=(self.num_nodes, self.embedding_dim),
		  initializer="uniform",
		  trainable=True)
		self.covariance = self.add_weight(name='euclidean_covariance', 
		  shape=(self.num_nodes, self.embedding_dim),
		  initializer="ones",
		  trainable=True)
		super(HyperbolicEmbeddingLayerAsymmetric, self).build(input_shape)

	def call(self, idx):

		source_embedding = tf.gather(self.embedding, idx[:,:1])
		target_embedding = tf.gather(self.embedding, idx[:,1:])

		source_covariance = tf.gather(self.covariance, idx[:,:1])
		# make covariance diagonal 
		source_covariance_inv = 1. / source_covariance * self.identity

		source_covariance_det = K.prod(source_covariance, axis=-1, keepdims=False)

		dist = mobius_add(target_embedding, -source_embedding)
		# dist = target_embedding - source_embedding

		sigma_inv_dist  = mobius_matrix_vector_multiplication(source_covariance_inv, dist)
		# sigma_inv_dist  = K.batch_dot(dist, source_covariance_inv, axes=[2,1])

		# sigma_inv_dist = poincare_ball_to_hyperboloid(sigma_inv_dist)
		# dist = poincare_ball_to_hyperboloid(dist)

		dd = K.sqrt((2 * np.pi) ** self.embedding_dim * source_covariance_det)

		probs = K.exp( - 0.5 * ((K.sum(dist[...,:-1] * sigma_inv_dist[...,:-1], 
							axis=-1) + dist[...,-1] * sigma_inv_dist[...,-1])))  / dd

		probs = probs / (K.sum(probs, axis=-1, keepdims=True) + K.epsilon())

		return probs


	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1] - 1)
	
	def get_config(self):
		base_config = super(HyperbolicEmbeddingLayerAsymmetric, self).get_config()
		base_config.update({"num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim, })
		return base_config

def build_poincare_asym_model(num_nodes, embedding_dim, num_negative_samples, 
	euc_lr=1e-3, hyp_lr=1e-3):
	x = Input((1 + 1 + num_negative_samples, ),
		dtype=tf.int32)
	poincare_embedding = HyperbolicEmbeddingLayerAsymmetric(num_nodes, embedding_dim)(x)

	model = Model(x, poincare_embedding)

	optimizer = PoincareOptimizer(euc_lr=euc_lr, hyp_lr=hyp_lr)

	model.compile(optimizer=optimizer, 
		loss="sparse_categorical_crossentropy",
		target_tensors=[ tf.placeholder(dtype=tf.int32), ])

	return model

class HyperbolicLogisticLayer(Layer):

	def __init__(self,
		units,
		**kwargs):
		super(HyperbolicLogisticLayer, self).__init__(**kwargs)
		self.units = units
		self.zero = K.zeros((1, self.units))

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.W = self.add_weight(name='euclidean_weight', 
		  shape=( self.units, input_shape[-1]),
		  initializer="uniform",
		  regularizer=regularizers.l2(1e-10),
		  trainable=True) # euclidean weight

		self.bias = self.add_weight(name='poincare_bias', 
		  shape=(self.units, input_shape[-1]),
		  initializer="uniform",
		  trainable=True) # hyperbolic (poincare) weight

		super(HyperbolicLogisticLayer, self).build(input_shape)

	def call(self, x):

		if len(x.shape) == 3:
			nsamples = x.shape[1]
			x = K.reshape(x, [-1, x.shape[-1]])
		else:
			nsamples = None

		W = self.W
		bias = self.bias

		bias = K.expand_dims(bias, 0)

		x = K.expand_dims(x, 1)
		x = K.tile(x, [1, self.units, 1]) # repeat embedding for each class

		a_k = parallel_transport(W, bias)

		lambdas = lambda_x(bias)
		a_k_norm = norm(a_k)

		minus_p_k_plus_x = mobius_add(-bias, x)

		uu = 2 * K.sqrt(c) * K.sum(minus_p_k_plus_x * a_k, axis=-1, keepdims=True)

		dd = (1 - c * norm(minus_p_k_plus_x) ** 2 ) * a_k_norm
		unnormalized_probs = K.exp(lambdas * a_k_norm / K.sqrt(c) * tf.asinh( uu / dd )) 

		unnormalized_probs = K.squeeze(unnormalized_probs, -1)
		normalized_probs =  unnormalized_probs / K.sum(unnormalized_probs, axis=-1, keepdims=True)

		if nsamples is not None:
			normalized_probs = K.reshape(normalized_probs, [-1, nsamples, self.units])

		return normalized_probs

	# def norm(self, x, axis=-1, keepdims=True):
	# 	return K.sqrt(K.sum(K.square(x), axis=axis, keepdims=keepdims))

	# def mobius_add(self, x, y):
	# 	c = self.c
	# 	xy = K.sum(x * y, axis=-1, keepdims=True)
	# 	norm_x = self.norm(x, keepdims=True)
	# 	norm_y = self.norm(y, keepdims=True)
	# 	dd = 1 + 2 * c * xy + c ** 2 * norm_x ** 2 * norm_y ** 2
	# 	uu = (1 + 2 * c * xy + c * norm_y ** 2) * x + (1 - c * norm_x ** 2) * y
	# 	return uu / dd

	# def lambda_x(self, x, axis=-1):
	# 	c = self.c
	# 	norm_x = self.norm(x, axis=axis)
	# 	return 2 / (1 - c * norm_x ** 2)

	# def parallel_transport(self, v, x):
	# 	return self.lambda_x(self.zero) / self.lambda_x(x) * v

	def compute_output_shape(self, input_shape):
		return tuple(list(input_shape)[:-1] + [self.units])
	
	def get_config(self):
		base_config = super(HyperbolicLogisticLayer, self).get_config()
		base_config.update({"units": self.units, "c": self.c})
		return base_config

def build_supervised_poincare_model(num_nodes, embedding_dim, num_negative_samples, 
	num_classes, euc_lr=1e-3, hyp_lr=1e-3):
	
	x = Input((1 + 1 + num_negative_samples, ),
		dtype=tf.int32)
	poincare_embedding = PoincareEmbeddingLayer(num_nodes, embedding_dim)(x)

	hyperboloid_embedding = Lambda(poincare_ball_to_hyperboloid)(poincare_embedding)
	y = HyperbolicLogisticLayer(num_classes)(poincare_embedding)
	model = Model(x, #hyperboloid_embedding)
		[hyperboloid_embedding, y], )

	optimizer = PoincareOptimizer(euc_lr=euc_lr, hyp_lr=hyp_lr)

	model.compile(optimizer=optimizer, 
		# loss=hyperbolic_softmax_loss(sigma=1.),
		# target_tensors=[ tf.placeholder(dtype=tf.int32), ])
		loss=[hyperbolic_softmax_loss(sigma=1.), "sparse_categorical_crossentropy"],
		loss_weights=[1e-0, 1e-2],
		target_tensors=[ tf.placeholder(dtype=tf.int32),  tf.placeholder(dtype=tf.int32)] )

	return model


class PoincareOptimizer(optimizer.Optimizer):
	
	def __init__(self, 
		euc_lr=1e-3,
		hyp_lr=1e-1, 
		use_locking=False,
		name="PoincareOptimizer"):
		super(PoincareOptimizer, self).__init__(use_locking, name)
		self.euc_lr = euc_lr
		self.hyp_lr = hyp_lr

	def _apply_dense(self, grad, var):
		if "poincare" in var.name:
			# poincare update
			grad = 1 / lambda_x(var) ** 2 * grad
			exp_map = self.poincare_exponential_map(- self.hyp_lr * grad, var, )
			return tf.assign(var, exp_map)
		else:
			# euclidean update
			return tf.assign_add(var, - self.euc_lr * grad)

	def _apply_sparse(self, grad, var):

		indices = grad.indices
		values = grad.values

		var_ = tf.gather(var, indices, name="gather_apply_sparse")

		if "poincare" in var.name:
			# poincare update
			grad = 1 / lambda_x(var_) ** 2 * values
			exp_map = self.poincare_exponential_map( -self.hyp_lr * values, var_, )
			exp_map = var_
			return tf.scatter_update(ref=var, indices=indices, updates=exp_map, name="scatter_update")
		else:
			# euclidean update
			values = K.zeros_like(values)
			return tf.scatter_add(ref=var, indices=indices, updates= - self.euc_lr * values)

	def poincare_exponential_map(self, v, x):
		l_x = lambda_x(x)
		norm_v = norm(v)

		cosh_lambda_norm_v = tf.cosh(l_x * norm_v)
		sinh_lambda_norm_v = tf.sinh(l_x * norm_v)
		x_dot_unit_v = K.batch_dot(x, v / norm_v, axes=1)

		u1 = l_x * (cosh_lambda_norm_v + x_dot_unit_v * sinh_lambda_norm_v)

		u2 = 1 / norm_v * sinh_lambda_norm_v

		d = 1 + (l_x - 1) * cosh_lambda_norm_v + l_x * x_dot_unit_v * sinh_lambda_norm_v

		return u1 / d * x + u2 / d * v

	
def squash(x):
	norm_x = np.linalg.norm(x, axis=-1, keepdims=True)
	return norm_x ** 2 / (1 + norm_x ** 2) * x / norm_x

import matplotlib.pyplot as plt

def build_hyperbolic_logistic_regression_model(input_dim, output_dim, euc_lr=1e-3, hyp_lr=1e-1):

	x = Input(shape=(input_dim,))
	y = HyperbolicLogisticLayer(output_dim, name="hyperbolic_logistic_layer")(x)

	model = Model(x, y)

	opt = PoincareOptimizer(euc_lr=euc_lr, hyp_lr=hyp_lr) 

	model.compile(optimizer=opt, loss="categorical_crossentropy")

	return model

# def parallel_transport(v, x):	
# 	zero = np.zeros_like(x)
# 	return lambda_x(zero) / lambda_x(x) * v

# def mobius_add(x, y):
# 	c = 1
# 	norm_x = np.linalg.norm(x, axis=-1, keepdims=True)
# 	norm_y =  np.linalg.norm(y, axis=-1, keepdims=True)

# 	xy = x.dot(y.T)

# 	uu = (1 + 2 * c * xy + c * norm_y ** 2) * x + (1 - c * norm_x ** 2) * y
# 	dd = 1 + 2 * c * xy + c ** 2 * norm_x ** 2 * norm_y ** 2
# 	return uu / dd

def main():

	np.random.seed(0)
	tf.set_random_seed(0)

	from sklearn.datasets import make_classification
	from sklearn.preprocessing import OneHotEncoder

	X, Y = make_classification(n_samples=1000, n_classes=4, n_clusters_per_class=1,  
		n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=0)
	labels  = OneHotEncoder().fit_transform(Y[:,None]).A

	poincare_embedding = squash(2*X)

	model = build_hyperbolic_logistic_regression_model(poincare_embedding.shape[1], 
		labels.shape[1],
		euc_lr=1e-3, 
		hyp_lr=1e-2)

	model.summary()

	from sklearn.model_selection import StratifiedShuffleSplit

	sss = StratifiedShuffleSplit(n_splits=1, 
		test_size=.2, 
		random_state=0)
	split_train, split_test = next(sss.split(poincare_embedding, labels.argmax(-1)))

	from keras.callbacks import EarlyStopping

	model.fit(poincare_embedding[split_train], labels[split_train],
		epochs=1000, 
		verbose=1,
		validation_split=.1,
		callbacks=[EarlyStopping(patience=1000)])

	predictions = model.predict(poincare_embedding[split_test])

	from sklearn.metrics import f1_score

	print (labels[split_test].argmax(-1))
	print (predictions.argmax(-1))

	print ("f1 score", f1_score(labels[split_test].argmax(-1), predictions.argmax(-1), average="micro"))

	weights, biases = model.get_weights()

	a_prime = parallel_transport(weights, biases)

	assert (np.linalg.norm(biases, axis=-1) < 1).all()

	hyperplanes = [[] for _ in range(len(weights))]

	for x in np.arange(-1, 1, 0.01):
		for y in np.arange(-1, 1, 0.01,):
			x_ = np.array([[x, y]])
			if np.linalg.norm(x_) >= 1:
				continue

			m_add = mobius_add(-biases, x_)

			y = (m_add * a_prime).sum(-1)

			idx, = np.where(np.abs(y) < 1e-3)
			for i in idx:
				hyperplanes[i].append(x_)

	hyperplanes = [np.concatenate(hyperplane, axis=0) for hyperplane in hyperplanes]

	plt.figure(figsize=[10, 10])

	num_labels = len(set(Y))
	colors = np.random.rand(num_labels, 3)

	plt.scatter(poincare_embedding[:,0], poincare_embedding[:,1], c=colors[Y], s=5)
	for i, hyperplane in enumerate(hyperplanes):
		plt.scatter(hyperplane[:,0], hyperplane[:,1], c=colors[i][None,:])
	plt.show()

	plt.figure(figsize=[10, 10])
	plt.scatter(poincare_embedding[split_test,0], poincare_embedding[split_test,1], c=colors[predictions.argmax(-1)], s=5)
	for i, hyperplane in enumerate(hyperplanes):
		plt.scatter(hyperplane[:,0], hyperplane[:,1], c=colors[i][None,:])
	plt.show()
	

if __name__ == "__main__":
	main()