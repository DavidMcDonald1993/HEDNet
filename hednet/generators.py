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
		graph,
		passes=100,
		):
		assert isinstance(positive_samples, np.ndarray)
		assert isinstance(negative_samples, list)
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]
		self.negative_samples = negative_samples
		self.num_positive_samples = len(positive_samples)
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model
		self.graph = graph
		self.N = len(graph)
		self.context_size = args.context_size
		
		# self.sps = nx.floyd_warshall_numpy(graph, 
		# 	nodelist=sorted(graph))

		self.passes = passes

		print ("Built generator")

		# N = 5000
		# pairs = np.ones((N, N))
		# for row in positive_samples:
		# 	pairs[row[0], row[1]] = 0

		# for i in range(10000):
		# 	u, v = np.unravel_index(
		# 		np.searchsorted(negative_samples[1], 
		# 		np.random.rand(self.num_negative_samples*self.batch_size)),
		# 		shape=(N, N))

		# 	pairs[u, v] = 0

		# 	print (i, pairs.sum())

		# 	if pairs.sum() == 0:
		# 		print ("DONE", i)
		# 		break

		# raise SystemExit

	def __len__(self):
		return 10000
		# return int(np.ceil(len(self.positive_samples) /\
		# 	float(self.batch_size))) * self.passes

	def __getitem__(self, batch_idx):

		batch_size = self.batch_size
		positive_samples = self.positive_samples
		num_negative_samples = self.num_negative_samples
		negative_samples = self.negative_samples
		N = self.N
		num_positive_samples = self.num_positive_samples
		# passes = self.passes

		# batch_idx = batch_idx % passes
	
		# batch_positive_samples = positive_samples[
		# 	batch_idx * batch_size : \
		# 	(batch_idx+1) * batch_size
		# ]

		# idx = random.choices(
		# 	range(len(positive_samples)), 
		# 	k=batch_size)
		idx = np.random.choice(num_positive_samples, 
			size=batch_size)
		batch_positive_samples = positive_samples[idx]

		assert self.context_size == 1
		# idx = batch_positive_samples[:,-1].argsort()
		# batch_positive_samples = batch_positive_samples[idx]
		
		# assert np.allclose(negative_samples[1][-1], 1)
		# assert negative_samples[1][-1] == 1

		# batch_negative_samples = np.concatenate([
		# 	np.unravel_index(np.searchsorted(negative_samples[k], 
		# 		np.random.rand(count*num_negative_samples)),
		# 		dims=(N, N))
		# 	for k, count in Counter(batch_positive_samples[:,-1]).items()
		# ], axis=1).T

		# batch_negative_samples = np.column_stack(
		# 	np.unravel_index(np.searchsorted(negative_samples[1], 
		# 		np.random.rand(batch_size * num_negative_samples)),
		# 		shape=(N, N)), )
		batch_negative_samples = np.searchsorted(negative_samples[1],
			np.random.rand(batch_size * num_negative_samples, 2))
		# batch_negative_samples = np.random.randint(self.N, 
		# 	size=(batch_size * num_negative_samples, 2))


		batch_positive_samples = np.expand_dims(
			batch_positive_samples[:,:-1], axis=1)
		batch_negative_samples = batch_negative_samples.reshape(
			batch_size, num_negative_samples, 2)

		training_sample = np.concatenate(
			(batch_positive_samples, batch_negative_samples), 
			axis=1).reshape(batch_size*(num_negative_samples+1), 2)

		# training_sample = np.empty(
		# 	(len(batch_positive_samples) + len(batch_negative_samples), 2), 
		# 	)
		# training_sample[::num_negative_samples+1] = batch_positive_samples[:,:-1]
		# for i in range(num_negative_samples):
		# 	training_sample[i+1::num_negative_samples+1] = \
		# 		batch_negative_samples[i::num_negative_samples]

		# for i in range(self.batch_size):

		# 	arr = training_sample[i*(num_negative_samples+1) : \
		# 		(i+1)*(num_negative_samples+1)]
		# 	for row in arr[1:]:
		# 		assert self.sps[arr[0,0], arr[0,1]] < self.sps[row[0], row[1]]


		# training_sample = np.array([
		# 	np.append(sample[:-1], 
		# 		np.searchsorted(negative_samples[sample[2]]\
		# 		[sample[0]], np.random.rand(num_negative_samples)))
		# 	for sample in batch_positive_samples
		# ])

		target = np.zeros((len(training_sample), 1))

		return training_sample, target

	def on_epoch_end(self):
		positive_samples = self.positive_samples
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]
		pass