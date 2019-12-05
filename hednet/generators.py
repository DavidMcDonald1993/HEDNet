from __future__ import print_function

import random
import numpy as np
import scipy as sp
import networkx as nx

from keras.utils import Sequence

import itertools

from collections import Counter

class TrainingDataGenerator(Sequence):

	def __init__(self,
		positive_samples,
		negative_samples,
		model,
		args,
		graph
		):
		assert isinstance(positive_samples, np.ndarray)
		assert isinstance(negative_samples, list)
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]
		self.negative_samples = negative_samples
	
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model
		self.graph = graph
		self.N = len(graph)
		self.context_size = args.context_size
		
		# self.sps = nx.floyd_warshall_numpy(graph, nodelist=sorted(graph))

		print ("Built generator")

	# def get_training_sample(self, batch_positive_samples):
	# 	negative_samples = self.negative_samples
	# 	num_negative_samples = self.num_negative_samples

	# 	batch_negative_samples = np.array([
	# 		# samples +
	# 		# random.choices(self.negative_samples[samples[0]],
	# 		# 	k=num_negative_samples-(len(samples)-1))
	# 		# for samples in batch_positive_samples
	# 		np.searchsorted(negative_samples[k][u], 
	# 			np.random.rand(num_negative_samples))
	# 		for u, _, k in batch_positive_samples
	# 	],
	# 	dtype=np.int64)


	# 	batch_nodes = np.concatenate([batch_positive_samples[:,:-1],
	# 		batch_negative_samples], axis=1)

	# 	# batch_nodes = batch_negative_samples

	# 	return batch_nodes

	def __len__(self):
		return 1000
		# return int(np.ceil(len(self.positive_samples) / float(self.batch_size)))

	def __getitem__(self, batch_idx):

		batch_size = self.batch_size
		positive_samples = self.positive_samples
		num_negative_samples = self.num_negative_samples
		negative_samples = self.negative_samples
		N = self.N
	
		idx = random.choices(range(len(positive_samples)), 
			k=batch_size)
		batch_positive_samples = positive_samples[idx]

		idx = batch_positive_samples[:,-1].argsort()
		batch_positive_samples = batch_positive_samples[idx]

		assert np.allclose(negative_samples[1][-1], 1)
		assert negative_samples[1][-1] == 1

		batch_negative_samples = np.concatenate([
			np.unravel_index(np.searchsorted(negative_samples[k], 
			[random.random() 
				for _ in range(count*num_negative_samples)]),
				shape=(N, N))
			# np.unravel_index(np.searchsorted(negative_samples[k], 
				# np.random.rand(count*num_negative_samples)),
				# dims=(N, N))
			for k, count in Counter(batch_positive_samples[:,-1]).items()
		], axis=1).T

		
		training_sample = np.empty(
			(len(batch_positive_samples) + len(batch_negative_samples), 2), dtype=int)
		training_sample[::num_negative_samples+1] = batch_positive_samples[:,:-1]
		for i in range(num_negative_samples):
			training_sample[i+1::num_negative_samples+1] = batch_negative_samples[i::num_negative_samples]

		# for i in range(self.batch_size):

		# 	arr = training_sample[i*(num_negative_samples+1) : (i+1)*(num_negative_samples+1)]
		# 	for row in arr[1:]:
		# 		assert self.sps[arr[0,0], arr[0,1]] < self.sps[row[0], row[1]]



		target = np.zeros((len(training_sample), 1))

		return training_sample, target

	def on_epoch_end(self):
		pass