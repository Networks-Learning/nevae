from utils import create_dir, pickle_save, print_vars, load_data
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
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        #'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'edges': tf.placeholder((tf.int32, tf.int32)),
        'non_edges': tf.placeholder((tf.int32, tf.int32))
    }
def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_epochs", type=int, default=32, help="Number of epochs")
    parser.add_argument("--graph_file", type=str, default=None,
                        help="The file where the graph structure is saved")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      graph_file=flags.graph_file,
      out_dir=flags.out_dir,
      #training
      num_epochs=flags.num_epochs
      )

if __name__ == '__main__':
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    hparams = create_hparams(FLAGS)

    # loading the data from a file
    adj, features = load_data(hparams.graph_file)
    num_nodes = adj[0].shape
    num_features = features[0].shape

    model = VAEG(placeholders, num_nodes, num_features)
    model.initialize()
    model.train()
    '''
    Test code
    model2 = VRNN(True)
    model2.restore()
    print(model2.sample())
    '''
