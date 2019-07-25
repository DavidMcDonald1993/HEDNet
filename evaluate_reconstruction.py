import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"


import numpy as np
import networkx as nx
import pandas as pd

import argparse

from aheat.utils import load_embedding, load_data

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances

import functools
import fcntl

import glob


def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape) 
	return np.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def hyperbolic_distance_hyperboloid(u, v):
	u = np.expand_dims(u, axis=1)
	v = np.expand_dims(v, axis=0)
	mink_dp = -minkowski_dot(u, v)
	mink_dp = np.maximum(mink_dp - 1, 1e-15)
	return np.squeeze(np.arccosh(1 + mink_dp), axis=-1)

def hyperbolic_distance_poincare(X):
	norm_X = np.linalg.norm(X, keepdims=True, axis=-1)
	norm_X = np.minimum(norm_X, np.nextafter(1,0, ))
	uu = euclidean_distances(X) ** 2
	dd = (1 - norm_X**2) * (1 - norm_X**2).T
	return np.arccosh(1 + 2 * uu / dd)

def logarithmic_map(p, x):

	alpha = -minkowski_dot(p, x)

	alpha = np.maximum(alpha, 1+1e-15)

	return np.arccosh(alpha) * (x - alpha * p) / \
		np.sqrt(alpha ** 2 - 1) 
		

def parallel_transport(p, q, x):
	assert len(p.shape) == len(q.shape) == len(x.shape)
	alpha = -minkowski_dot(p, q)
	return x + minkowski_dot(q - alpha * p, x) * (p + q) / \
		(alpha + 1) 

# def log_likelihood(x, mu, sigma_inv, sigma_det):

# 	# ignore zero t coordinate
# 	x = x[...,:-1]

# 	k = x.shape[-1]

# 	x_minus_mu = x - mu

# 	uu = np.sum(x_minus_mu ** 2 * sigma_inv, 
# 		axis=-1, keepdims=True) # assume sigma inv is diagonal

# 	return - 0.5 * (np.log(np.maximum(sigma_det, 1e-15)) +\
# 		uu + k * np.log(2 * np.pi))

# def hyperbolic_log_pdf(mus, sigmas):
	

# 	dim = mus.shape[1] - 1

# 	# project to tangent space
# 	source_mus = np.expand_dims(mus, axis=1)
# 	target_mus = np.expand_dims(mus, axis=0)

# 	to_tangent_space = logarithmic_map(source_mus, 
# 		target_mus)

# 	# parallel transport to mu zero
# 	mu_zero = np.zeros((1, 1, dim + 1))
# 	mu_zero[..., -1] = 1

# 	to_tangent_space_mu_zero = parallel_transport(source_mus,
# 	 mu_zero, to_tangent_space)

# 	# compute euclidean_log_pdf

# 	source_sigmas = np.expand_dims(sigmas, axis=0)
# 	sigma_inv = 1 / source_sigmas
# 	sigma_det = np.prod(source_sigmas, axis=-1, keepdims=True) 

# 	logs = log_likelihood(to_tangent_space_mu_zero, 
# 		np.zeros((1, 1, dim)), 
# 		sigma_inv, 
# 		sigma_det)
# 	logs = np.squeeze(logs, axis=-1)

# 	# compute log det proj v

# 	norm = np.sqrt(np.maximum(0.,
# 		minkowski_dot(to_tangent_space_mu_zero,
# 			to_tangent_space_mu_zero)))
# 	norm = np.squeeze(norm, axis=-1)

# 	log_det_proj = (dim - 0) * (np.log(np.maximum(np.sinh(norm), 1e-15)) -\
# 		np.log(np.maximum(norm, 1e-15)))
	
# 	return logs - log_det_proj

def kullback_leibler_divergence(mus, sigmas):

	dim = mus.shape[1] - 1

	# project to tangent space
	source_mus = np.expand_dims(mus, axis=1)
	target_mus = np.expand_dims(mus, axis=0)

	to_tangent_space = logarithmic_map(source_mus, 
		target_mus)

	# dists = hyperbolic_distance_hyperboloid(source_mus,
	# 	target_mus)
	# dists = np.squeeze(dists, 1)

	# parallel transport to mu zero
	mu_zero = np.zeros((1, 1, dim + 1))
	mu_zero[..., -1] = 1
	
	to_tangent_space_mu_zero = parallel_transport(source_mus,
		mu_zero, 
		to_tangent_space)

	# sigmas = np.maximum(sigmas, 1e-15)

	source_sigmas = np.expand_dims(sigmas, axis=1)
	target_sigmas = np.expand_dims(sigmas, axis=0)

	# mu is zero vector
	# ignore zero t coordinate
	x_minus_mu = to_tangent_space_mu_zero[...,:-1]

	trace = np.sum(target_sigmas / \
		source_sigmas, 
		axis=-1, keepdims=True)

	uu = np.sum(x_minus_mu ** 2 / \
		source_sigmas, 
		axis=-1, keepdims=True) # assume sigma is diagonal

	log_det = np.sum(np.log(target_sigmas), 
		axis=-1, keepdims=True) - \
		np.sum(np.log(source_sigmas), 
		axis=-1, keepdims=True)

	return 0.5 * (trace + uu - dim - log_det) 

def evaluate_rank_and_MAP(scores, 
	edgelist, non_edgelist):
	assert not isinstance(edgelist, dict)
	assert (scores <= 0).all()

	if not isinstance(edgelist, np.ndarray):
		edgelist = np.array(edgelist)

	if not isinstance(non_edgelist, np.ndarray):
		non_edgelist = np.array(non_edgelist)

	edge_scores = scores[edgelist[:,0], edgelist[:,1]]
	non_edge_scores = scores[non_edgelist[:,0], non_edgelist[:,1]]

	labels = np.append(np.ones_like(edge_scores), 
		np.zeros_like(non_edge_scores))
	scores_ = np.append(edge_scores, non_edge_scores)
	ap_score = average_precision_score(labels, scores_) # macro by default
	auc_score = roc_auc_score(labels, scores_)
		
	# fpr, tpr, thresholds = roc_curve(labels, scores_)
	# import matplotlib.pyplot as plt

	# plt.figure()
	# lw = 2
	# plt.plot(fpr, tpr, color='darkorange',
	# 		lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
	# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver operating characteristic')
	# plt.legend(loc="lower right")
	# plt.show()

	idx = (-non_edge_scores).argsort()
	ranks = np.searchsorted(-non_edge_scores, 
		-edge_scores, sorter=idx) + 1
	ranks = ranks.mean()

	print ("MEAN RANK =", ranks, "AP =", ap_score, 
		"AUROC =", auc_score)

	return ranks, ap_score, auc_score

def touch(path):
	with open(path, 'a'):
		os.utime(path, None)

def read_edgelist(fn):
	edges = []
	with open(fn, "r") as f:
		for line in (l.rstrip() for l in f.readlines()):
			edge = tuple(int(i) for i in line.split("\t"))
			edges.append(edge)
	return edges

def lock_method(lock_filename):
	''' Use an OS lock such that a method can only be called once at a time. '''

	def decorator(func):

		@functools.wraps(func)
		def lock_and_run_method(*args, **kwargs):

			# Hold program if it is already running 
			# Snippet based on
			# http://linux.byexamples.com/archives/494/how-can-i-avoid-running-a-python-script-multiple-times-implement-file-locking/
			fp = open(lock_filename, 'r+')
			done = False
			while not done:
				try:
					fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
					done = True
				except IOError:
					pass
			return func(*args, **kwargs)

		return lock_and_run_method

	return decorator 

def threadsafe_fn(lock_filename, fn, *args, **kwargs ):
	lock_method(lock_filename)(fn)(*args, **kwargs)

def save_test_results(filename, seed, data, ):
	d = pd.DataFrame(index=[seed], data=data)
	if os.path.exists(filename):
		test_df = pd.read_csv(filename, sep=",", index_col=0)
		test_df = d.combine_first(test_df)
	else:
		test_df = d
	test_df.to_csv(filename, sep=",")

def threadsafe_save_test_results(lock_filename, filename, seed, data):
	threadsafe_fn(lock_filename, save_test_results, filename=filename, seed=seed, data=data)


def parse_args():

	parser = argparse.ArgumentParser(description='Load Hyperboloid Embeddings and evaluate reconstruction')
	
	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--embedding", dest="embedding_directory",  
		help="directory of embedding to load.")

	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--poincare", action="store_true")

	return parser.parse_args()

def elu(x, alpha=1.):
	x = x.copy()
	mask = x <= 0
	x[mask] = alpha * (np.exp(x[mask]) - 1)
	return x

def main():

	args = parse_args()

	args.directed = True

	graph, features, node_labels = load_data(args)
	assert nx.is_directed(graph)
	print ("Loaded dataset")
	print ()

	test_edges = np.array(list(graph.edges()))
	test_non_edges = np.array(list(nx.non_edges(graph)))

	# np.random.seed(args.seed)
	# idx = np.random.permutation(np.arange(len(test_non_edges), dtype=int))[:len(test_edges)]
	# test_non_edges = test_non_edges[idx]

	files = sorted(glob.iglob(os.path.join(args.embedding_directory, 
		"*.csv")))
	embedding_filename, variance_filename = files[-2:]
	# embedding_filename = files[-1]

	print ("loading embedding from", embedding_filename)
	embedding_df = load_embedding(embedding_filename)
	embedding_df = embedding_df.reindex(sorted(embedding_df.index))
	# row 0 is embedding for node 0
	# row 1 is embedding for node 1 etc...
	embedding = embedding_df.values

	dists = hyperbolic_distance_hyperboloid(embedding,
		embedding)

	print ("DISTANCE")
	evaluate_rank_and_MAP(-dists, 
		test_edges, test_non_edges)

	print ("loading variance from", variance_filename)
	variance_df = pd.read_csv(variance_filename, index_col=0)
	variance_df = variance_df.reindex(sorted(variance_df.index))
	variance = variance_df.values

	print (variance.min(), variance.max())
	variance = elu(variance, alpha=1-1e-7) + 1
	print (variance.min(), variance.max())

	scores = -kullback_leibler_divergence(embedding, variance)
	scores = np.squeeze(scores, axis=-1)

	print ()
	print ("DIRECTION")
	print (np.mean([scores[u, v] > scores[v, u] for u, v in graph.edges() if (v, u) not in graph.edges()]))
	print()
	# (mean_rank_recon, ap_recon, 
	# roc_recon) = evaluate_rank_and_MAP(scores, 
	# 	[(u, v) for u, v in graph.edges if (v, u) not in graph.edges], 
	# 	[(v, u) for u, v in graph.edges if (v, u) not in graph.edges])
	# print ()

	for u, v in np.random.permutation(test_edges)[:10]:
		print (u, v, scores[u, v])
	print()
	print (np.mean([scores[u, v] for u, v in test_edges]))
	print ()

	for u, v in np.random.permutation(test_non_edges)[:10]:
		print (u, v, scores[u, v])
	print ()
	print (np.mean([scores[u, v] for u, v in test_non_edges]))
	print ()

	print ("KULLBACK LEIBLER")
	(mean_rank_recon, ap_recon, 
	roc_recon) = evaluate_rank_and_MAP(scores, 
		test_edges, test_non_edges)

	raise SystemExit

	test_results = dict()

	test_results.update({"mean_rank_recon": mean_rank_recon, 
		"ap_recon": ap_recon,
		"roc_recon": roc_recon})

	test_results_dir = args.test_results_dir
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir)
	test_results_filename = os.path.join(test_results_dir, "test_results.csv")
	test_results_lock_filename = os.path.join(test_results_dir, "test_results.lock")
	touch(test_results_lock_filename)

	print ("saving test results to {}".format(test_results_filename))

	threadsafe_save_test_results(test_results_lock_filename, test_results_filename, args.seed, data=test_results )

	print ("done")


if __name__ == "__main__":
	main()