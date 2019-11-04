import keras.backend as K
import tensorflow as tf 
from keras.layers import Input, Concatenate
from keras.models import Model

from hednet.losses import asym_hyperbolic_loss
from hednet.hyperboloid_layers import HyperboloidGaussianEmbeddingLayer
from hednet.optimizers import ExponentialMappingOptimizer, MyAdamOptimizer

def build_hyperboloid_asym_model(num_nodes, 
	embedding_dim, 
	context_size,
	num_negative_samples,
	batch_size,  
	lr=1e-1, ):
	x = Input((1 + 1 + 0, ),
		dtype=tf.int64)
	weights = Input(( 1, ),
		dtype=K.floatx())

	hyperboloid_embedding = HyperboloidGaussianEmbeddingLayer(num_nodes, embedding_dim)(x)

	hyperboloid_embedding = Concatenate()([hyperboloid_embedding, weights])

	model = Model([x, weights], hyperboloid_embedding)

	optimizer = ExponentialMappingOptimizer(lr=lr,)
	# optimizer = MyAdamOptimizer()

	model.compile(optimizer=optimizer, 
		loss=asym_hyperbolic_loss(num_negative_samples),
		target_tensors=[ tf.placeholder(dtype=tf.int64), ])

	return model