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
		# probs,
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
		# self.positive_samples = positive_samples
		
		# self.nodes = [n
		# 	for n in positive_samples
		# 	if len(positive_samples[n][0]) > 0]
			# and len(positive_samples[n][1]) > 0]

		# assert np.allclose(probs[:,-1], 1)
		# self.probs = probs
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model
		self.graph = graph
		self.N = len(graph)
		self.context_size = args.context_size
		# self.negative_samples = {n: set() for n in positive_samples}
		# for n, row in enumerate(probs):
		# 	if probs[n, 0] > 0:
		# 		self.negative_samples[n].add(0)
		# 	for i in range(1, len(row)):
		# 		if probs[n, i-1] < probs[n, i]:
		# 			self.negative_samples[n].add(i)
		# self.negative_samples = {n: list(l)
		# 	for n, l in self.negative_samples.items()}
		# for n, l in self.negative_samples.items():
		# 	self.positive_samples[n][args.context_size] = l
		
		# self.context_size = args.context_size + 1

		# self.weights = {
		# 	n : { i: float(len(positive_samples[n][i]) )
		# 	for i in range(self.context_size) }
		# 	for n in self.positive_samples
		# }
		# self.inx = {n: 
		# 	np.array([i for i in range(self.context_size)
		# 	if self.weights[n][i] > 0] )
		# 	for n in self.positive_samples
		# }

		# self.choices = {
		# 	i : [n for n, l in self.inx.items()
		# 		if i in l]
		# 	for i in range(self.context_size)
		# }

		# print ("computing shortest path lengths")
		# self.sps = nx.floyd_warshall_numpy(graph,
		# 	nodelist=sorted(graph))

		# for n in self.positive_samples:
		# 	for i in range(self.context_size):
		# 		for v in self.positive_samples[n][i]:
		# 			assert self.sps[n, v] == i+1
		# raise SystemExit

		print ("Built generator")

	def get_training_sample(self, batch_positive_samples):
		negative_samples = self.negative_samples
		num_negative_samples = self.num_negative_samples

		batch_negative_samples = np.array([
			# samples +
			# random.choices(self.negative_samples[samples[0]],
			# 	k=num_negative_samples-(len(samples)-1))
			# for samples in batch_positive_samples
			np.searchsorted(negative_samples[k][u], 
				np.random.rand(num_negative_samples))
			for u, _, k in batch_positive_samples
		],
		dtype=np.int64)


		batch_nodes = np.concatenate([batch_positive_samples[:,:-1],
			batch_negative_samples], axis=1)

		# batch_nodes = batch_negative_samples

		return batch_nodes

	def __len__(self):
		return 1000
		# return int(np.ceil(len(self.positive_samples) / float(self.batch_size)))

	def __getitem__(self, batch_idx):

		# np.random.seed(batch_idx)

		batch_size = self.batch_size
		positive_samples = self.positive_samples
		num_negative_samples = self.num_negative_samples
		negative_samples = self.negative_samples
		N = self.N
		# inx = self.inx
		# choices = self.choices

		###################################################

		# # # nodes = self.nodes[batch_idx * batch_size : (batch_idx + 1) * batch_size]
		# nodes = random.choices(self.nodes, k=batch_size)
		# nodes = self.nodes
		# nodes = random.choices(self.nodes, k=batch_size*(1+num_negative_samples))
		# pos_nodes = nodes[::num_negative_samples+1]

		#####################################################

		# batch_positive_samples = positive_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
		idx = random.choices(range(len(positive_samples)), 
			k=batch_size)
		batch_positive_samples = positive_samples[idx]

		idx = batch_positive_samples[:,-1].argsort()
		batch_positive_samples = batch_positive_samples[idx]


		# training_sample = self.get_training_sample(batch_positive_samples)


		# target = np.zeros((len(training_sample), 1))

		# weights = np.ones_like(target)
		# assert (weights > 0).all()

		# return [training_sample, weights], target


		#####################################################

		# for k, count in Counter(batch_positive_samples[:,-1]).items():
		# 	print (k, count)
		# print ()
		# raise SystemExit


		batch_negative_samples = np.concatenate([
			np.unravel_index(np.searchsorted(negative_samples[k], 
			[random.random() 
				for _ in range(count*num_negative_samples)]),
				shape=(N, N))
		# 	# np.unravel_index(np.searchsorted(negative_samples[k], 
		# 		# np.random.rand(count*num_negative_samples)),
		# 		# dims=(N, N))
			for k, count in Counter(batch_positive_samples[:,-1]).items()
		], axis=1).T

		# batch_negative_samples = np.array([ 
		# 	(u, np.searchsorted(negative_samples[k][u], random.random()))
		# 	for u, _, k in batch_positive_samples
		# 	for _ in range(num_negative_samples)
		# ])


		# training_sample = []
		# for i in range(batch):
		# 	training_sample.append(pos.pop(0))
		# 	training_sample += neg[:num_negative_samples]
		# 	neg = neg[num_negative_samples:]

		# training_sample = np.array(training_sample)
		# training_sample = np.concatenate([
		# 	batch_positive_samples[i,:-1],
		# 	batch_negative_samples[i*num_negative_samples:(i+1)*num_negative_samples]
		# 	for i in range(batch_size)
		# ])

		# print (training_sample.shape)
		# raise SystemExit

		training_sample = np.empty(
			(len(batch_positive_samples) + len(batch_negative_samples), 2), dtype=int)
		training_sample[::num_negative_samples+1] = batch_positive_samples[:,:-1]
		for i in range(num_negative_samples):
			training_sample[i+1::num_negative_samples+1] = batch_negative_samples[i::num_negative_samples]

		# for i in range(batch_size):
		# 	arr = training_sample[i*(num_negative_samples+1):
		# 		(i+1)*(num_negative_samples+1)]
		# 	assert not np.allclose(arr[0,0], arr[0,1])
		# 	for a in arr[1:]:
		# 		print (self.sps[arr[0, 0], arr[0, 1]], self.sps[a[0], a[1]], )
		# 		assert (self.sps[arr[0, 0], arr[0, 1]] <
		# 			self.sps[a[0], a[1]])
		# print ()
		# raise SystemExit

		target = np.zeros((len(training_sample), 1))

		weights = np.ones_like(target)
		assert (weights > 0).all()

		return [training_sample, weights], target

		###################################################

		# neighbors = [
		# 	random.choice(range(len(inx[n]) - 1))
		# 	# random.choice(list(set(range(self.context_size-1)).\
		# 	# 	intersection(inx[n]))) 
		# 	for n in nodes
		# ]

		# l = [
		# 	random.choice(range(i+1, self.context_size))
		# 	for i in neighbors
		# 	for _ in range(num_negative_samples)
		# ]

		# n_ = [
		# 	random.choice(choices[i])
		# 	for i in l
		# ]

		# pos = [
		# 	(n, random.choice(positive_samples[n][i]))
		# 	for n, i in zip(nodes, neighbors)
		# ]

		# neg = [
		# 	(n, random.choice(positive_samples[n][i]))
		# 	for n, i in zip(n_, l)
		# ]

		# training_sample = []
		# while len(pos) > 0:
		# 	training_sample.append(pos.pop(0))
		# 	training_sample += neg[:num_negative_samples]
		# 	neg = neg[num_negative_samples:]

		# training_sample = np.array(training_sample)

		# for i in range(batch_size):
		# 	arr = training_sample[i*11:(i+1)*11]
		# 	for a in arr[1:]:
		# 		assert (self.sps[arr[0, 0], arr[0, 0]] <
		# 			self.sps[a[0], a[1]])

		# training_sample = np.array([
		# 	(n_, random.choice(positive_samples[n_][l]))
		# 	if 
		# 	n_ == n
		# 	else 
		# 	(n_, random.choice(positive_samples[n_][l+1]))
		# 	for n, l in zip(nodes, neighbors)
		# 	for n_ in [n] + 
		# 		random.choices(choices[l+1], 
		# 		k=num_negative_samples)
		# ])

		training_sample = np.array([ 
			(nodes[batch_idx*(num_negative_samples+1)+i], 
				random.choice(positive_samples[nodes[batch_idx*(num_negative_samples+1)+i]][k]))
			for batch_idx, pos_k in enumerate((random.choice(inx[n][:-1]) for n in pos_nodes))
			for i, k in enumerate([pos_k] + [random.choice(inx[n][inx[n]>pos_k]) 
				for n in nodes[batch_idx*(num_negative_samples+1)+1:(batch_idx+1)*(num_negative_samples+1)]])
		])


		# for i in range(batch_size):
		# 	arr = training_sample[i*(num_negative_samples+1):(i+1)*(num_negative_samples+1)]
		# 	assert not np.allclose(arr[0,0], arr[0,1])
		# 	for a in arr[1:]:
		# 		# print (self.sps[arr[0, 0], arr[0, 1]], self.sps[a[0], a[1]])
		# 		assert (self.sps[arr[0, 0], arr[0, 1]] <
		# 			self.sps[a[0], a[1]])
			# print ()
		# raise SystemExit

		target = np.zeros((len(training_sample), 1))

		weights = np.ones_like(target)
		assert (weights > 0).all()

		return [training_sample, weights], target

		###########################################################


		neighbors = [
			self.inx[n][random.choice(range(len(self.inx[n])-1)):] 
			for n in nodes
		]

		training_sample = []
		for n, neighbor in zip(nodes, neighbors):
			sample = [n] +\
				random.sample(self.positive_samples[n][neighbor[0]], k=1)
			for l in neighbor[1:]:
				if len(sample) == 2 + self.num_negative_samples:
					break
				if len(sample) + len(self.positive_samples[n][l]) <= 2 + self.num_negative_samples:
					sample += self.positive_samples[n][l]

				else:
					k = 2 + self.num_negative_samples - len(sample)
					sample += random.sample(self.positive_samples[n][l], k=k)
					
			training_sample.append(sample)

		training_sample = np.array(training_sample)

		target = np.zeros((len(training_sample), 1))

		weights = np.ones_like(target)
		assert (weights > 0).all()

		return [training_sample, weights], target

		#####################################################

		# neighbors = [
		# 	self.inx[n] for n in nodes
		# ]

		# samples = [
		# 	[random.sample(self.positive_samples[n][k], k=1)[0]
		# 	for k in neighbor]
		# 	for n, neighbor in zip(nodes, neighbors)
		# ]
		
		# training_sample = np.array([
		# 	[u, v, w] 
		# 	for u, sample in zip(nodes, samples)
		# 	for v, w in itertools.combinations(sample, 2)
		# ])

		# # # for u, v, w in training_sample:
		# # # 	assert self.sps[u, v] < self.sps[u, w]

		# weights = np.array([
		# 	[self.weights[n][i] * self.weights[n][j]]
		# 	for n in nodes
		# 	for i, j in itertools.combinations(self.inx[n], 2)
		# ])
		# # weights = np.ones((len(training_sample), 1))
		# # weights /= weights.sum()

		# assert len(training_sample) == len(weights)

		# target = np.zeros((len(training_sample), 1))

		# return [training_sample, weights], target

		###################################################
		batch_positive_samples = positive_samples[
			batch_idx * batch_size : (batch_idx + 1) * batch_size]
		training_sample = self.get_training_sample(
			batch_positive_samples)

		# for row in training_sample:
		# 	u = row[0]
		# 	assert u not in row[1:]

		# training_sample = random.sample(self.triplets, 
		# 	self.batch_size)
		# training_sample = np.array(training_sample)

		# for i, row in enumerate(training_sample):
			# assert tuple(row) in self.triplets
			# u, v = row[:2]
			# for v_ in row[2:]:
			# 	assert self.sps[u, v] < self.sps[u, v_]
		
		target = np.zeros((len(training_sample), 1))
		# target = np.zeros((len(nodes), 1, ),
		# 	dtype=np.int64)
		# target = np.ones((len(nodes), 1, ),
		# 	dtype=np.float64)

		weights = np.ones_like(target)
		# weights = np.array([
		# 	[self.weights[n][neighbor[0]] * \
		# 	self.weights[n][neighbor[1]]
		# 	]
		# 	for n, neighbor in zip(nodes, neighbors)])
		# weights /= weights.sum()
		assert (weights > 0).all()

		return [training_sample, weights], target

	def on_epoch_end(self):
		pass
		# self.nodes = np.random.permutation(self.nodes)

		# positive_samples = self.positive_samples
		# idx = np.random.permutation(len(positive_samples))
		# self.positive_samples = positive_samples[idx]