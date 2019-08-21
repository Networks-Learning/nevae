from utils import *
#create_dir, pickle_save, print_vars, load_data, get_shape, proxy
from config import SAVE_DIR, VAEGConfig
from datetime import datetime
from cell import VAEGCell
from model import VAEG

import tensorflow as tf
import numpy as np
import logging
import pickle
import os
import argparse

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FLAGS = None
placeholders = {
    'dropout': tf.placeholder_with_default(0., shape=()),
    'lr': tf.placeholder_with_default(0., shape=()),
    'decay': tf.placeholder_with_default(0., shape=())
    }
def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_epochs", type=int, default=32, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--decay_rate", type=float, default=1.0, help="decay rate")
    parser.add_argument("--dropout_rate", type=float, default=0.00005, help="dropout rate")
    parser.add_argument("--log_every", type=int, default=5, help="write the log in how many iterations")
    parser.add_argument("--sample_file", type=str, default=None, help="directory to store the sample graphs")

    parser.add_argument("--random_walk", type=int, default=5, help="random walk depth")
    parser.add_argument("--z_dim", type=int, default=5, help="z_dim")
    parser.add_argument("--nodes", type=int, default=5, help="z_dim")
    parser.add_argument("--bin_dim", type=int, default=3, help="bin_dim")

    parser.add_argument("--graph_file", type=str, default=None,
                        help="The dictory where the training graph structure is saved")
    parser.add_argument("--z_dir", type=str, default=None,
                        help="The z values will be stored file to be stored")
    parser.add_argument("--sample", type=bool, default=False, help="True if you want to sample")
    parser.add_argument("--mask_weight", type=bool, default=False, help="True if you want to mask weight")
    parser.add_argument("--neg_sample_size", type=int, default=10, help="number of negative edges to be sampled")
    parser.add_argument("--node_sample", type=int, default=1, help="number of nodes to be sampled")
    parser.add_argument("--bfs_sample", type=int, default=1, help="number of times bfs to run")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      graph_file=flags.graph_file,
      out_dir=flags.out_dir,
      z_dir=flags.z_dir,
      sample_file=flags.sample_file,
      z_dim=flags.z_dim,

      # training
      learning_rate=flags.learning_rate,
      decay_rate=flags.decay_rate,
      dropout_rate=flags.dropout_rate,
      num_epochs=flags.num_epochs,
      random_walk=flags.random_walk,
      log_every=flags.log_every,
      nodes=flags.nodes,
      bin_dim = flags.bin_dim,
      mask_weight = flags.mask_weight,
      
      #sample
      sample=flags.sample,
      neg_sample_size=flags.neg_sample_size,
      node_sample=flags.node_sample,
      bfs_sample=flags.bfs_sample
      )

if __name__ == '__main__':
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    hparams = create_hparams(FLAGS)
    
    # loading the data from a file
    adj, weight, weight_bin, features, edges, neg_edges, features1, = load_data_new(hparams.graph_file, hparams.nodes, hparams.node_sample, hparams.bfs_sample, hparams.bin_dim)
    num_nodes = adj[0].shape[0]
    num_features = features[0].shape[1]
    lenedges = [len(edge[0]) for edge in edges]
    lenweight_bin = [len(weight_b[0]) for weight_b in weight_bin]
    print("Len edges", lenedges, lenweight_bin)
    print("Num features", num_features)
    print("Num examples", len(adj))
    #print("Neg_index", neg_index) 
    e = max([len(edge) for edge in edges])
        
    log_fact_k = log_fact(e)
    # Training
    #'''
    model = VAEG(hparams, placeholders, num_nodes, num_features,log_fact_k, len(adj))
    model.restore(hparams.out_dir)
    model.train(placeholders, hparams, adj, weight, weight_bin, features, edges, neg_edges, features1)
