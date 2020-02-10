from __future__ import print_function

import os
import re
import glob
import numpy as np
import pandas as pd

from keras.callbacks import Callback

from hednet.utils import hyperboloid_to_poincare_ball

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape) 
	return np.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def hyperbolic_distance_hyperboloid(x, y):
	mink_dp = -minkowski_dot(x, y)
	mink_dp = np.maximum(mink_dp, 1+1e-15)
	return np.arccosh(mink_dp)

def elu(x, alpha=1.):
	x = x.copy()
	mask = x <= 0
	x[mask] = alpha * (np.exp(x[mask]) - 1)
	return x

class Checkpointer(Callback):

	def __init__(self, 
		epoch,
		nodes,
		embedding_directory,
		history=0
		):
		self.epoch = epoch
		self.nodes = nodes
		self.embedding_directory = embedding_directory
		self.history = history


	def on_epoch_end(self, batch, logs={}):
		self.epoch += 1
		if self.epoch % 1 != 0:
			return
		print ("Epoch {} complete".format(self.epoch)) 
		self.remove_old_models()
		self.save_model()

	def remove_old_models(self):
		embedding_directory = self.embedding_directory
		# for old_model_path in sorted(
		# 	glob.iglob(os.path.join(self.embedding_directory, 
		# 		"[0-9]+_model.h5"))):
		for old_model_path in filter(
			re.compile("[0-9]+\_model\.h5").match, 
			os.listdir(embedding_directory)):
			print ("removing model: {}".format(old_model_path))
			os.remove(os.path.join(embedding_directory, 
				old_model_path))

	def save_model(self):

		weights_filename = os.path.join(self.embedding_directory,
			"{:05d}_model.h5".format(self.epoch))
		self.model.save_weights(weights_filename)
		print ("saving weights to", weights_filename)

		weights = self.model.get_weights()

		# embedding_filename = os.path.join(self.embedding_directory, 
		# 	"{:05d}_embedding.csv.gz".format(self.epoch))
		embedding = weights[0]
		embedding = hyperboloid_to_poincare_ball(embedding)
		print ("embedding", np.linalg.norm(embedding.mean(0)))
		ranks = np.linalg.norm(embedding, axis=-1)
		print ("ranks", ranks.min(), ranks.mean(),
			ranks.max() )
		# assert not np.any(np.isnan(embedding))
		# assert not np.any(np.isinf(embedding))
		# assert (embedding[:,-1] > 0).all(), embedding[:,-1]
		# assert np.allclose(minkowski_dot(embedding, embedding), -1)

		# print ("saving current embedding to {}".\
		# 	format(embedding_filename))
		# embedding_df = pd.DataFrame(embedding, index=self.nodes)
		# embedding_df.to_csv(embedding_filename, compression="gzip")

		# variance_filename = os.path.join(self.embedding_directory, 
		# 	"{:05d}_variance.csv.gz".format(self.epoch))
		variance = weights[1]
		variance = elu(variance) + 1
		print ("variance", variance.min(),
			variance.mean(), variance.max())

		# print ("saving current variance to {}".\
		# 	format(variance_filename))
		# variance_df = pd.DataFrame(variance, index=self.nodes)
		# variance_df.to_csv(variance_filename, compression="gzip")
		# print()
