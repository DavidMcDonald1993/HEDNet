import os

import random
import numpy as np
import networkx as nx

import argparse

from hednet.utils import load_data
from evaluation_utils import check_complete, load_embedding, compute_scores, evaluate_rank_AUROC_AP, touch, threadsafe_save_test_results
from remove_utils import sample_non_edges

def parse_args():

	parser = argparse.ArgumentParser(description='Load Embeddings and evaluate reconstruction')
	
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
	
	parser.add_argument("--dist_fn", dest="dist_fn", type=str,
		choices=["poincare", "hyperboloid", "euclidean", 
			"kle", "klh", "st"])

	return parser.parse_args()

def main():

	args = parse_args()

	test_results_dir = args.test_results_dir
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir, exist_ok=True)
	test_results_filename = os.path.join(test_results_dir, 
		"test_results.csv")

	if check_complete(test_results_filename, args.seed):
		return

	test_results_lock_filename = os.path.join(test_results_dir, 
		"test_results.lock")
	touch(test_results_lock_filename)

	args.directed = True

	graph, _ = load_data(args)
	assert nx.is_directed(graph)
	print ("Loaded dataset")
	print ()

	random.seed(args.seed)
	
	test_edges = list(graph.edges())
	num_edges = len(test_edges)

	test_non_edges = sample_non_edges(graph, 
		set(test_edges),
		num_edges)

	test_edges = np.array(test_edges)
	test_non_edges = np.array(test_non_edges)

	embedding = load_embedding(args.dist_fn, 
		args.embedding_directory)
	
	if isinstance(embedding, tuple):
		embedding, embedding_ = embedding
		print ("embedding shape is", embedding.shape)

		embedding_pos_u = embedding[test_edges[:,0]], embedding_[test_edges[:,0]]
		embedding_pos_v = embedding[test_edges[:,1]], embedding_[test_edges[:,1]]

		embedding_neg_u = embedding[test_non_edges[:,0]], embedding_[test_non_edges[:,0]]
		embedding_pneg_v = embedding[test_non_edges[:,1]], embedding_[test_non_edges[:,1]]

	else:

		print ("embedding shape is", embedding.shape)

		embedding_pos_u = embedding[test_edges[:,0]]
		embedding_pos_v = embedding[test_edges[:,1]]

		embedding_neg_u = embedding[test_non_edges[:,0]]
		embedding_neg_v = embedding[test_non_edges[:,1]]

	edge_scores = compute_scores(
		embedding_pos_u,
		embedding_pos_v, 
		args.dist_fn)

	non_edge_scores = compute_scores(
		embedding_neg_u,
		embedding_neg_v,
		args.dist_fn)

	test_results = dict()

	(mean_rank_recon, ap_recon, 
	roc_recon) = evaluate_rank_AUROC_AP(
		edge_scores, 
		non_edge_scores)

	test_results.update({"mean_rank_recon": mean_rank_recon, 
		"ap_recon": ap_recon,
		"roc_recon": roc_recon})

	# map_recon = evaluate_mean_average_precision(scores, 
	# 	test_edges)
	# test_results.update({"map_recon": map_recon})

	# precisions_at_k = [(k, 
	# 	evaluate_precision_at_k(scores,  
	# 		test_edges, k=k))
	# 		for k in (1, 3, 5, 10)]
	# for k, pk in precisions_at_k:
	# 	print ("precision at", k, pk)
	# test_results.update({"p@{}".format(k): pk
	# 	for k, pk in precisions_at_k})

	print ("saving test results to {}".format(test_results_filename))

	threadsafe_save_test_results(test_results_lock_filename, 
		test_results_filename, args.seed, data=test_results )

	print ("done")


if __name__ == "__main__":
	main()