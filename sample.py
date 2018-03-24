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
    parser.add_argument("--bin_dim", type=int, default=3, help="bin_dim")

    parser.add_argument("--graph_file", type=str, default=None,
                        help="The dictory where the training graph structure is saved")
    parser.add_argument("--z_dir", type=str, default=None,
                        help="The z values will be stored file to be stored")
    parser.add_argument("--sample", type=bool, default=False, help="True if you want to sample")

    parser.add_argument("--mask_weight", type=bool, default=False, help="True if you want to use masking")

    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")
    parser.add_argument("--edges", type=int, default=30, help="Number of edges to sample.")
    
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes to sample.")
    parser.add_argument("--offset", type=int, default=0, help="offset of sample.")



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
      mask_weight=flags.mask_weight,
      #sample
      sample=flags.sample,
      edges=flags.edges,
      nodes=flags.nodes,
      offset=flags.offset,
      bin_dim=flags.bin_dim
      )

if __name__ == '__main__':
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    hparams = create_hparams(FLAGS)
    
    # loading the data from a file
    adj, weight, weight_bin, features, edges = load_data(hparams.graph_file, hparams.nodes, hparams.bin_dim)
    
    #Test code
    #'''
    model2 = VAEG(hparams, placeholders, hparams.nodes, 1, edges)
    model2.restore(hparams.out_dir)
    #interpolation
    '''
    [i1, i2] =  [i for i in np.random.choice(len(adj), 2)]
    sample1 = model2.getembeddings(hparams, placeholders, adj[i1], features[i1],weight_bin[i1], weight[i1])
    sample2 = model2.getembeddings(hparams, placeholders, adj[i2], features[i2],weight_bin[i2], weight[i2])
    i = 0
    with open(hparams.sample_file + "inter/start_1.txt", "a") as f:
        f.write("#"+str(i1)+"\n")
        for (u,v,w) in edges[i1]:
                f.write(str(u)+" "+str(v)+" {\'weight\':"+str(w)+"}\n")
    with open(hparams.sample_file + "inter/start_2.txt", "a") as f:
        f.write("#"+str(i2)+"\n")
        for (u,v,w) in edges[i2]:
                f.write(str(u)+" "+str(v)+" {\'weight\':"+str(w)+"}\n")
    
    while i < 19:
        for index in range(1,20):
            model2.sample_graph_slerp(hparams, placeholders, i, sample1, sample2, 'lerp', (i+1)*0.05, index, num=11)
            model2.sample_graph_slerp(hparams, placeholders, i, sample1, sample2, 'slerp', (i+1)*0.05, index, num=11)
        i+=1
    
    '''
    #sample
    i = 0
    while i < 100 :
        model2.sample_graph_neighborhood(hparams, placeholders, adj, features, weight, weight_bin, i+hparams, hparams.node, (i+1) * 0.00001)
        #model2.sample_graph(hparams, placeholders,adj, features, weight, weight_bin, i+hparams.offset, hparams.nodes, hparams.edges)
        i += 1
    
