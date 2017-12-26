
from utils import create_dir, pickle_save, print_vars, load_data, get_shape
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
    parser.add_argument("--h_dim", type=int, default=5, help="hidden state RNN dimension")
    parser.add_argument("--z_dim", type=int, default=5, help="latent space dimension")
    parser.add_argument("--x_dim", type=int, default=5, help="input batches")

    parser.add_argument("--random_walk", type=int, default=5, help="random walk depth")

    parser.add_argument("--graph_file", type=str, default=None,
                        help="The directory where the training graph structure is saved")

    parser.add_argument("--z_dir", type=str, default=None,
                        help="The z values will be stored file to be stored")
    parser.add_argument("--sample", type=bool, default=False, help="True if you want to sample")
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

      # training
      learning_rate=flags.learning_rate,
      decay_rate=flags.decay_rate,
      dropout_rate=flags.dropout_rate,
      num_epochs=flags.num_epochs,
      random_walk=flags.random_walk,
      log_every=flags.log_every,

      #sample
      sample=flags.sample
      )

if __name__ == '__main__':
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    hparams = create_hparams(FLAGS)
    # loading the data from a file
    adj, features, edges = load_data(hparams.graph_file)
    num_nodes = adj[0].shape[0]
    num_features = features[0].shape[1]
    # Training
    model = VAEG(hparams, placeholders, num_nodes, num_features, edges)
    model.initialize()
    model.train(placeholders, hparams, adj, features)
    
    #Test code
    '''
    model2 = VAEG(hparams, placeholders, 10, 1)
    model2.restore(hparams.out_dir)
    hparams.sample = True
    i = 0
    while i < 5:
        model2.sample_graph(hparams, placeholders, i, 16)
        i += 1
    model2.plot_hspace(hparams, placeholders, 10)    
    '''
