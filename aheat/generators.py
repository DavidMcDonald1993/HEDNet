from __future__ import print_function

import random
import numpy as np
import scipy as sp
import networkx as nx

from keras.utils import Sequence

import itertools

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
		
		self.nodes = [n
			for n in positive_samples
			if len(positive_samples[n][0]) > 0]
			# and len(positive_samples[n][1]) > 0]

		# assert np.allclose(probs[:,-1], 1)
		self.probs = probs
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model
		self.graph = graph
		self.context_size = args.context_size
		self.negative_samples = {n: set() for n in positive_samples}
		for n, row in enumerate(probs):
			if probs[n, 0] > 0:
				self.negative_samples[n].add(0)
			for i in range(1, len(row)):
				if probs[n, i-1] < probs[n, i]:
					self.negative_samples[n].add(i)
		self.negative_samples = {n: list(l)
			for n, l in self.negative_samples.items()}
		for n, l in self.negative_samples.items():
			self.positive_samples[n][args.context_size] = l
		self.context_size = args.context_size + 1

		self.weights = {
			n : { i: float(len(positive_samples[n][i]) )
			for i in range(self.context_size) }
			for n in self.positive_samples
		}
		self.inx = {n: 
			[i for i in range(self.context_size)
			if self.weights[n][i] > 0] 
			for n in self.positive_samples
		}

		self.choices = {
			i : [n for n, l in self.inx.items()
			if i in l]
			for i in range(self.context_size)
		}

		# self.sps = nx.floyd_warshall_numpy(graph,
		# 	nodelist=sorted(graph))

	def get_training_sample(self, batch_positive_samples):
		num_negative_samples = self.num_negative_samples
		probs = self.probs

		source_nodes = batch_positive_samples[:,0]

		batch_negative_samples = np.array([
			# samples +
			# random.choices(self.negative_samples[samples[0]],
			# 	k=num_negative_samples-(len(samples)-1))
			# for samples in batch_positive_samples
			np.searchsorted(probs[u], 
				np.random.rand(num_negative_samples))
			for u in source_nodes
		],
		dtype=np.int64)


		batch_nodes = np.concatenate([batch_positive_samples,
			batch_negative_samples], axis=1)
		# batch_nodes = batch_negative_samples

		return batch_nodes

	def __len__(self):
		return 10000
		# return int(np.ceil(len(self.positive_samples) / float(self.batch_size)))

	def __getitem__(self, batch_idx):
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		num_negative_samples = self.num_negative_samples
		inx = self.inx
		choices = self.choices

		###################################################

		# # # nodes = self.nodes[batch_idx * batch_size : (batch_idx + 1) * batch_size]
		nodes = random.choices(self.nodes, k=batch_size)
		# nodes = self.nodes

		neighbors = [
			inx[n][random.choice(range(len(inx[n])-1))] 
			for n in nodes
		]

		training_sample = np.array([
			(n_, random.choice(positive_samples[n_][l]))
			if 
			n_ == n
			else 
			(n_, random.choice(positive_samples[n_][l+1]))
			for n, l in zip(nodes, neighbors)
			for n_ in [n] + 
				random.choices(choices[l+1], 
				k=num_negative_samples)
		])

		# for i in range(batch_size):
		# 	arr = training_sample[i*11:(1+i)*11]
		# 	u, v = arr[0]
		# 	for w, x in arr[1:]:
		# 		assert self.sps[u, v] < self.sps[w, x]
		# raise SystemExit

		# training_sample = []

		# for n, l in zip(nodes, neighbors):
		# 	training_sample.append([n] + \
		# 		random.sample(self.positive_samples[n][l], k=1))

		# 	for _ in range(self.num_negative_samples):
		# 		n_ = random.sample(self.choices[l+1] - {n}, k=1)
		# 		training_sample.append(n_ + \
		# 			random.sample(self.positive_samples[n_][l+1], k=1))

		# training_sample = np.array(training_sample)

		# print (training_sample.shape)
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