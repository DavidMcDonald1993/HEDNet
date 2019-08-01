from __future__ import print_function

import random
import numpy as np
import scipy as sp
import networkx as nx

from keras.utils import Sequence

import itertools

# def hyperbolic_distance(u, v):
	# mink_dp = u[:,:-1].dot(v[:,:-1].T) - u[:,-1:] * v[:,-1:].T
	# mink_dp = np.maximum(-mink_dp, 1 + 1e-15)
	# return np.arccosh(mink_dp)

class TrainingDataGenerator(Sequence):

	def __init__(self, 
		positive_samples, 
		probs, 
		model, 
		args,
		graph
		):
		# idx = np.random.permutation(len(positive_samples))
		# self.positive_samples = positive_samples[idx]
		self.positive_samples = positive_samples
		self.nodes = np.random.permutation(range(len(positive_samples)))
		assert np.allclose(probs[:,-1], 1)
		self.probs = probs
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model
		self.graph = graph
		self.negative_samples = {n: set() for n in positive_samples}
		for n, row in enumerate(probs):
			if probs[n, 0] > 0:
				self.negative_samples[n].add(0)
			for i in range(1, len(row)):
				if probs[n, i-1] < probs[n, i]:
					self.negative_samples[n].add(i)
		self.negative_samples = {n:list(l) for n, l in self.negative_samples.items()}

		self.context_size = args.context_size

		self.mask = np.zeros((len(positive_samples), self.context_size))
		for n in positive_samples:
			for i in range(self.context_size):
				if False:#len(positive_samples[n][i]) == 1 and n in positive_samples[n][i]:
					x = 0
					# self.positive_samples[n][i] = [np.random.choice(list(set(positive_samples) - {n}))]
				else:
					self.mask[n, i] = len(positive_samples[n][i]) 

		# embedding = self.model.get_weights()[0]
		# dists = hyperbolic_distance(embedding, embedding)
		# probs = np.exp(-dists) * self.probs_
		# probs /= probs.sum(axis=-1, keepdims=True)
		# self.probs = probs.cumsum(-1)

		# print ("computing sps")
		# import os
		# fn = "sps"
		# if not os.path.exists(fn):
		# 	sps = nx.floyd_warshall_numpy(graph)
		# 	np.savetxt(fn, sps)
		# else:
		# 	print ("loading")
		# 	sps = np.loadtxt(fn)
		# print ("done")

		# map_ = {n: i for i, n in enumerate(graph)}

		# idx = np.array([map_[n] for n in range(len(graph))])

		# sps = sps[idx]
		# sps = sps[:,idx]
		# self.sps=sps

	def get_training_sample(self, batch_positive_samples):
		num_negative_samples = self.num_negative_samples
		probs = self.probs

		source_nodes = batch_positive_samples[:,0]

		batch_negative_samples = np.array([
			random.choices(self.negative_samples[u], k=num_negative_samples)
			# np.searchsorted(probs[u], np.random.rand(num_negative_samples))
			for u in source_nodes
		], dtype=np.int64)

		# for (u, v), neg_samples in zip(batch_positive_samples.tolist(), batch_negative_samples):
		# 	assert u != v, (u, v)
		# 	assert v in self.graph.neighbors(u)
		# 	for v_ in neg_samples:
		# 		assert u != v_, ("neg", u, v_)
		# 		assert v_ not in self.graph.neighbors(u)
		# print (batch_negative_samples)
		# raise SystemExit

		batch_nodes = np.concatenate([batch_positive_samples, 
			batch_negative_samples], axis=1)

		return batch_nodes

	def __len__(self):
		# return 10000
		return int(np.ceil(len(self.positive_samples) / float(self.batch_size)))

	def __getitem__(self, batch_idx):
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		# np.random.seed(batch_idx)
		
		nodes = self.nodes[batch_idx * batch_size : (batch_idx + 1) * batch_size]
		# nodes = np.random.choice(self.nodes, size=batch_size, replace=True)
		# nodes = random.choices(self.nodes, k=batch_size)

		batch_positive_samples = np.array([[n] + [
			# np.random.choice(positive_samples[n][k])
			random.sample(positive_samples[n][k], k=1)[0] 
			for k in range(self.context_size)
		] 
		for n in nodes])

		# batch_positive_samples = positive_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
		training_sample = self.get_training_sample(batch_positive_samples)
		
		target = np.zeros((training_sample.shape[0], 1, ), 
			dtype=np.int64)

		# for i in range(len(training_sample)):
		# 	u = training_sample[i, 0]
		# 	for j, v in enumerate(training_sample[i, 1:]):
		# 		if j < 3:
		# 			assert self.sps[u, v] == j+1 or u==v
		# 		else:
		# 			assert self.sps[u, v] > 3

		mask = np.ones((training_sample.shape[0], training_sample.shape[1]-1))
		for row, n in enumerate(training_sample[:,0]):
			for i in range(self.context_size):
				mask[row, i] = self.mask[n, i]
			# assert n not in training_sample[row, 1:]

		# mask_ = np.concatenate([mask[:,i:i+1] * mask[:,j:j+1]
		# for i, j in itertools.combinations(range(3), 2)]
		# + [mask[:,i:i+1] * mask[:,j:j+1]
		# for i, j in itertools.product(range(3), range(3, 3+0))]
		# , axis=-1)

		# mask_ /= mask_.sum()
		# print (mask_)
		# raise SystemExit

		# assert (mask_ > 0).any(-1).all()

		return [training_sample, mask], target

	def on_epoch_end(self):
		pass
		# self.nodes = np.random.permutation(self.nodes)

		# positive_samples = self.positive_samples
		# idx = np.random.permutation(len(positive_samples))
		# self.positive_samples = positive_samples[idx]

		# embedding = self.model.get_weights()[-1]
		# dists = hyperbolic_distance(embedding, embedding)
		# probs = np.exp(-dists) * self.probs_
		# probs /= probs.sum(axis=-1, keepdims=True)
		# self.probs = probs.cumsum(-1)