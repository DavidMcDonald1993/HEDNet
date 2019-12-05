from __future__ import print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import random
import numpy as np
import pandas as pd
import glob

from keras import backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping

import tensorflow as tf

from hednet.utils import hyperboloid_to_poincare_ball, load_data
from hednet.utils import determine_positive_and_negative_samples
from hednet.generators import TrainingDataGenerator
from hednet.visualise import draw_graph, plot_degree_dist
from hednet.callbacks import Checkpointer, elu
from hednet.models import build_hyperboloid_asym_model
from hednet.optimizers import ExponentialMappingOptimizer
from hednet.losses import asym_hyperbolic_loss

K.set_floatx("float64")
# K.set_epsilon(np.float64(1e-15))

np.set_printoptions(suppress=True)

# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

config.log_device_placement=False
config.allow_soft_placement=True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

def load_weights(model, args):

	previous_models = sorted(glob.glob(os.path.join(args.embedding_path, "*.h5")))
	if len(previous_models) > 0:
		weight_file = previous_models[-1]
		initial_epoch = int(weight_file.split("/")[-1].split("_")[0])
		print ("previous models found in directory -- loading from file {} and resuming from epoch {}".format(weight_file, initial_epoch))
		model.load_weights(weight_file)
	else:
		print ("no previous model found in {}".format(args.embedding_path))
		initial_epoch = 0

	return model, initial_epoch

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(
		description="HEDNET algorithm for feature learning on directed complex networks")

	parser.add_argument("--edgelist", dest="edgelist", type=str, default=None,
		help="edgelist to load.")
	parser.add_argument("--labels", dest="labels", type=str, default=None,
		help="path to labels (just used for plotting)")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")
	parser.add_argument("--lr", dest="lr", type=np.float64, default=.1,
		help="Learning rate (default is .1).")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=50,
		help="The number of epochs to train for (default is 50).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=512, 
		help="Batch size for training (default is 512).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context-size", dest="context_size", type=int, default=1,
		help="Context size for generating positive samples (default is 1).")
	parser.add_argument("--patience", dest="patience", type=int, default=5,
		help="The number of epochs of no improvement in loss before training is stopped. (Default is 5)")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 10).", default=10)

	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", 
		help="Use this flag to set verbosity of training.")
	parser.add_argument('--workers', dest="workers", type=int, default=2, 
		help="Number of worker threads to generate training patterns (default is 2).")

	parser.add_argument("--embedding", dest="embedding_path", default=None, 
		help="path to save embedings.")

	parser.add_argument('--visualise', action="store_true", 
		help='flag to visualise embedding (embedding_dim must be 2)')

	args = parser.parse_args()
	return args

def configure_paths(args):
	'''
	build directories on local system for output of model after each epoch
	'''
	if not os.path.exists(args.embedding_path):
		os.makedirs(args.embedding_path)
		print ("making {}".format(args.embedding_path))
	print ("saving embedding to {}".format(args.embedding_path))

def main():

	args = parse_args()

	assert not (args.visualise and args.embedding_dim > 2), "Can only visualise two dimensions"
	assert args.embedding_path is not None, "you must specify a path to save embedding"

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	graph, node_labels = load_data(args)
	if not args.visualise and node_labels is not None:
		node_labels = None
	print ("Loaded dataset")

	if False:
		plot_degree_dist(graph, "degree distribution")

	configure_paths(args)
	print ("Configured paths")

	# build model
	num_nodes = len(graph)
	
	model = build_hyperboloid_asym_model(num_nodes, 
		args.embedding_dim, 
		args.context_size,
		args.num_negative_samples, 
		args.batch_size,
		lr=args.lr)
	model, initial_epoch = load_weights(model, args)

	model.summary()

	callbacks = [
		TerminateOnNaN(),
		EarlyStopping(monitor="loss", 
			patience=args.patience, 
			mode="min",
			verbose=True),
		Checkpointer(epoch=initial_epoch, 
			nodes=sorted(graph.nodes()), 
			# history=args.patience,
			embedding_directory=args.embedding_path)
	]			

	positive_samples, negative_samples = \
			determine_positive_and_negative_samples(graph, 
			args)

	print ("Training with data generator with {} worker threads".format(args.workers))
	training_generator = TrainingDataGenerator(positive_samples,  
		negative_samples,
		model,
		args,
		graph
	)

	model.fit_generator(training_generator, 
		workers=args.workers,
		max_queue_size=100, 
		# use_multiprocessing=args.workers>0, 
		use_multiprocessing=False,
		epochs=args.num_epochs, 
		steps_per_epoch=len(training_generator),
		initial_epoch=initial_epoch, 
		verbose=args.verbose,
		callbacks=callbacks
	)

	print ("Training complete")

	print ("saving final embedding")

	embedding_filename = os.path.join(args.embedding_path, 
			"final_embedding.csv.gz")
	embedding = model.get_weights()[0]
	embedding_df = pd.DataFrame(embedding, index=sorted(graph.nodes()))
	embedding_df.to_csv(embedding_filename)

	print ("saved final embedding to {}".format(embedding_filename))

	variance_filename = os.path.join(args.embedding_path, 
		"final_variance.csv.gz")
	variance = model.get_weights()[1]

	variance = elu(variance) + 1

	print ("saving final variance to {}".format(variance_filename))
	variance_df = pd.DataFrame(variance, index=sorted(graph.nodes()))
	variance_df.to_csv(variance_filename)

	if args.visualise:
		embedding = model.get_weights()[0]
		if embedding.shape[1] == 3:
			print ("projecting to poincare ball")
			embedding = hyperboloid_to_poincare_ball(embedding)
		draw_graph(graph,
			embedding, 
			node_labels, 
			path="2d-poincare-disk-visualisation.png")

if __name__ == "__main__":
	main()