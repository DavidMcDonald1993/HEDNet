from __future__ import print_function

import re
import sys
import os
import glob
import numpy as np
import pandas as pd

from keras.callbacks import Callback

def minkowski_dot(u):
	return (u[:,:-1] ** 2).sum(axis=-1, keepdims=True) - u[:,-1:] ** 2


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
		history=3
		):
		self.epoch = epoch
		self.nodes = nodes
		self.embedding_directory = embedding_directory
		self.history = history

	def on_epoch_end(self, batch, logs={}):
		self.epoch += 1
		print ("Epoch {} complete".format(self.epoch)) 
		self.remove_old_models()
		self.save_model()

	def remove_old_models(self):
		for old_model_path in sorted(glob.glob(os.path.join(self.embedding_directory, "*.csv")))[:-2*self.history]:
			print ("removing model: {}".format(old_model_path))
			os.remove(old_model_path)

	def save_model(self):
		embedding_filename = os.path.join(self.embedding_directory, "{:05d}_embedding.csv".format(self.epoch))
		embedding = self.model.get_weights()[0]
		assert not np.any(np.isnan(embedding))
		assert not np.any(np.isinf(embedding))
		assert all(embedding[:,-1] > 0)
		assert (np.allclose(minkowski_dot(embedding), -1))
		print (embedding[:5,-5:])
		print ("min t:", embedding[:,-1].min(), "max t:", embedding[:,-1].max())
		print ("saving current embedding to {}".format(embedding_filename))

		embedding_df = pd.DataFrame(embedding, index=self.nodes)
		embedding_df.to_csv(embedding_filename)

		variance_filename = os.path.join(self.embedding_directory, "{:05d}_variance.csv".format(self.epoch))
		variance = self.model.get_weights()[1]

		variance_ = elu(variance) + 1

		print (variance_[:5][:,:5])
		print ("variance min:", variance_.min(), "variance max:", variance_.max())

		print ("saving current variance to {}".format(variance_filename))

		variance_df = pd.DataFrame(variance, index=self.nodes)
		variance_df.to_csv(variance_filename)