from __future__ import print_function

import os
import fcntl
import functools
import numpy as np
import networkx as nx

import random

# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import StandardScaler

import pandas as pd

import pickle as pickle


from multiprocessing.pool import Pool 

# import matplotlib.pyplot as plt

# from collections import Counter


def load_data(args):

	edgelist_filename = args.edgelist
	labels_filename = args.labels

	graph = nx.read_weighted_edgelist(edgelist_filename, delimiter="\t", nodetype=int,
		create_using=nx.DiGraph())

	# print ("removing self loop edges")
	# graph.remove_edges_from(nx.selfloop_edges(graph))

	zero_weight_edges = [(u, v) for u, v, w in graph.edges(data="weight") if w == 0.]
	print ("removing", len(zero_weight_edges), "edges with 0. weight")
	graph.remove_edges_from(zero_weight_edges)

	print ("ensuring all weights are positive")
	nx.set_edge_attributes(graph, name="weight", values={edge: abs(weight) 
		for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})

	# graph = max(nx.strongly_connected_component_subgraphs(graph), key=len)
	# graph = nx.convert_node_labels_to_integers(graph)

	print ("number of nodes: {}\nnumber of edges: {}\n".format(len(graph), len(graph.edges())))

	if labels_filename is not None:

		print ("loading labels from {}".format(labels_filename))

		if labels_filename.endswith(".csv") or labels_filename.endswith(".csv.gz"):
			labels = pd.read_csv(labels_filename, index_col=0, sep=",")
			labels = labels.reindex(sorted(graph.nodes())).values.astype(int)#.flatten()
			assert len(labels.shape) == 2
		elif labels_filename.endswith(".pkl"):
			with open(labels_filename, "rb") as f:
				labels = pkl.load(f)
			labels = np.array([labels[n] for n in sorted(graph.nodes())], dtype=np.int)
		else:
			raise Exception

		print ("labels shape is {}\n".format(labels.shape))

	else:
		labels = None

	return graph, labels

def load_embedding(embedding_filename):
	assert embedding_filename.endswith(".csv.gz")
	embedding_df = pd.read_csv(embedding_filename, index_col=0)
	return embedding_df

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def poincare_ball_to_hyperboloid(X):
	x = 2 * X
	t = 1. + np.sum(np.square(X), axis=-1, keepdims=True)
	x = np.concatenate([x, t], axis=-1)
	return 1 / (1. - np.sum(np.square(X), axis=-1, keepdims=True)) * x

# def alias_setup(probs):
# 	'''
# 	Compute utility lists for non-uniform sampling from discrete distributions.
# 	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
# 	for details
# 	'''

# 	n, probs = probs

# 	K = len(probs)
# 	q = np.zeros(K)
# 	J = np.zeros(K, dtype=np.int)

# 	smaller = []
# 	larger = []
# 	for kk, prob in enumerate(probs):
# 		q[kk] = K*prob
# 		if q[kk] < 1.0:
# 			smaller.append(kk)
# 		else:
# 			larger.append(kk)

# 	while len(smaller) > 0 and len(larger) > 0:
# 		small = smaller.pop()
# 		large = larger.pop()

# 		J[small] = large
# 		q[large] = q[large] + q[small] - 1.0
# 		if q[large] < 1.0:
# 			smaller.append(large)
# 		else:
# 			larger.append(large)

# 	return n, (J, q)

# def alias_draw(J, q, size=1):
# 	'''
# 	Draw sample from a non-uniform discrete distribution using alias sampling.
# 	'''
# 	K = len(J)

# 	kk = np.floor(np.random.uniform(high=K, size=size)).astype(np.int)
# 	r = np.random.uniform(size=size)
# 	idx = r >= q[kk]
# 	kk[idx] = J[kk[idx]]
# 	return kk

# def convert_edgelist_to_dict(edgelist, undirected=True, self_edges=False):
# 	if edgelist is None:
# 		return None
# 	if undirected:
# 		edgelist += [(v, u) for u, v in edgelist]
# 	edge_dict = {}
# 	for u, v in edgelist:
# 		if self_edges:
# 			default = set(u)
# 		else:
# 			default = set()
# 		edge_dict.setdefault(u, default).add(v)
# 	edge_dict = {k: list(v) for k, v in edge_dict.items()}

# 	return edge_dict

# def get_training_sample(samples, num_negative_samples, ):
# 	positive_sample_pair, probs = samples
# 	negative_samples_ = np.random.choice(len(probs), replace=True, size=num_negative_samples, p=probs)
# 	return np.append(positive_sample_pair, negative_samples_, )

# def build_training_samples(positive_samples, negative_samples, num_negative_samples, alias_dict):
# 	input_nodes = positive_samples[:,0]
# 	print ("Building training samples")
	
# 	with Pool(processes=2) as p:
# 		training_samples = p.map(functools.partial(get_training_sample,
# 			num_negative_samples=num_negative_samples,
# 			),
# 			zip(positive_samples,
# 				# (negative_samples[u] for u in input_nodes),
# 				(alias_dict[u] for u in input_nodes)))
# 	return np.array(training_samples)



def determine_positive_and_negative_samples(graph, args):


	def build_positive_samples(graph, k=3):
		from scipy.sparse import identity

		assert k > 0

		def step(X):
			X[X>0] = 1    
			X[X<0] = 0
			return X
		
		N = len(graph)
		A0 = identity(N, dtype=int)
		print ("determined 0 hop neighbours")
		A1 = step(nx.adjacency_matrix(graph, nodelist=sorted(graph)) - A0)
		print ("determined 1 hop neighbours")
		positive_samples = [A0, A1]
		for i in range(2, k+1):
			A_k = step(step(positive_samples[-1].dot(A1)) - step(np.sum(positive_samples, axis=0)))
			print ("determined", i, "hop neighbours")
			positive_samples.append(A_k)
		return positive_samples

	def positive_samples_to_list(positive_samples):
		l = []
		for k, ps in enumerate(positive_samples):
			if k == 0:
				continue
			nzx, nzy = np.nonzero(ps)
			l.append(np.array((nzx, nzy, [k]*len(nzx))))
		return np.concatenate(l, axis=1).T

	def build_negative_samples(positive_samples):
		
		N = positive_samples[0].shape[0]
		negative_samples = []

		counts = np.sum(positive_samples, axis=0).sum(axis=1).A
		assert np.all(counts > 0)
		for k in range(len(positive_samples)):
			if True or k == args.context_size:
				# neg_samples = counts * counts.T 
				# neg_samples = np.ones((N, N), ) #* np.exp(-(args.context_size+1))
				# neg_samples[
				# 		np.sum(positive_samples[:k+1], axis=0).nonzero()
				# 	] = 0
				# assert np.allclose(neg_samples.diagonal(), 0)
				neg_samples = counts **.75
			else:
				assert False
				neg_samples = np.zeros((N, N))
				neg_samples[np.sum(positive_samples[k+1:], axis=0).nonzero()] = 1
			
			neg_samples = neg_samples.flatten()

			neg_samples /= neg_samples.sum(axis=-1, keepdims=True)
			neg_samples = neg_samples.cumsum(axis=-1)
			assert np.allclose(neg_samples[..., -1], 1)
			neg_samples[np.abs(neg_samples - neg_samples.max()) < 1e-15] = 1 
			negative_samples.append(neg_samples)
		return negative_samples

	positive_samples = build_positive_samples(graph, k=args.context_size)

	negative_samples = build_negative_samples(positive_samples)

	positive_samples = positive_samples_to_list(positive_samples)

	print ("found {} positive sample pairs".format(len(positive_samples)))


	return positive_samples, negative_samples
