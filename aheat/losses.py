import numpy as np
import tensorflow as tf 
import keras.backend as K

import itertools

def minkowski_dot(x, y):
    assert len(x.shape) == len(y.shape)
    return K.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def asym_hyperbolic_loss(y_true, y_pred):
	num_pos = 1
	num_neg = 3

	y_pred, mask = y_pred[:,:num_neg+num_pos], y_pred[:,num_neg+num_pos:]

	# pred = K.concatenate([   
	# 	y_pred[:,i:i+1] + K.log(K.sum(K.exp(-y_pred[:,i:]), axis=-1, keepdims=True)) 
	# 	for i in range(num_pos)	
	# ])

	# pred = - K.log( tf.nn.sigmoid(K.concatenate([y_pred[:,j:j+1] - y_pred[:,i:i+1]
	# 	for i, j in itertools.combinations(range(num_pos), 2)]
	# 	+ [y_pred[:,j:j+1] - y_pred[:,i:i+1] 
	# 	for i, j in itertools.product(range(num_pos), range(num_pos, num_pos+num_neg))]
	# 	, axis=-1)) )

	# y_pred_pos = K.square(y_pred)
	# y_pred_neg = K.exp(-y_pred)
	# pred = K.concatenate([y_pred_pos[:,i:i+1] + y_pred_neg[:,j:j+1]
	# 	for i, j in itertools.combinations(range(num_pos), 2)]
	# 	+ [y_pred_pos[:,i:i+1] + y_pred_neg[:,j:j+1]
	# 	for i, j in itertools.product(range(num_pos), 
	# 	range(num_pos, num_pos + num_neg))], axis=-1)

	# mask = K.concatenate([mask[:,i:i+1] * mask[:,j:j+1]
	# 	for i, j in itertools.combinations(range(num_pos), 2)]
	# 	+ [mask[:,i:i+1] * mask[:,j:j+1]
	# 	for i, j in itertools.product(range(num_pos), 
	# 	range(num_pos, num_pos + num_neg))], axis=-1)

	pred = K.concatenate([tf.nn.sparse_softmax_cross_entropy_with_logits(\
		labels=y_true[:,0], 
		logits=-y_pred[:,i:], )[:,None]
		for i in range(num_pos)]
		, axis=-1)

	# return K.sum(mask * pred / (K.sum(mask) + 1e-7))
	# return K.mean(K.sum(mask * pred / (K.sum(mask, axis=-1, keepdims=True) + 1e-7), axis=-1, keepdims=False) )
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
