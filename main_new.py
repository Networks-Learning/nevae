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

    parser.add_argument("--graph_file", type=str, default=None,
                        help="The dictory where the training graph structure is saved")
    parser.add_argument("--z_dir", type=str, default=None,
                        help="The z values will be stored file to be stored")
    parser.add_argument("--sample", type=bool, default=False, help="True if you want to sample")

    parser.add_argument("--mask_weight", type=bool, default=False, help="True if we want to mask")

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
      offset=flags.offset
      )

if __name__ == '__main__':
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    hparams = create_hparams(FLAGS)
    
    # loading the data from a file
    adj, features, edges = load_data(hparams.graph_file, hparams.nodes)

    num_nodes = adj[0].shape[0]
    #print adj[-1]
    #for i in range(len(adj)):
    #    print adj[i].shape
    #num_features = features[0].shape[1]
    
    # Training
    #model = VAEG(hparams, placeholders, num_nodes, num_features)
    #model.initialize()
    #model.train(placeholders, hparams, adj, features)
    
    #Test code
    #'''
    
    model2 = VAEG(hparams, placeholders, hparams.nodes, 1, edges)
    model2.restore(hparams.out_dir)
    #hparams.sample = True
    
    i = 0
    '''
    for i in range(len(adj)):
        sample_1 = model2.getembeddings(hparams, placeholders, adj[i], features[i])
        with open(hparams.z_dir+"sparse_"+str(i+1)+'.txt', 'a') as f:
            for z_i in sample_1:
                f.write('['+','.join([str(el[0]) for el in z_i])+']\n')
    '''
    '''
    sample_1 = model2.getembeddings(hparams, placeholders, adj[0], features[0]) 
    sample_2 = model2.getembeddings(hparams, placeholders, adj[1], features[1])
   
    '''
    #model2.get_stat(hparams, placeholders, hparams.nodes)
    #model2.zspace_analysis(hparams, placeholders)
    #G_good = load_embeddings(hparams.z_dir+'train0.txt')
    #G_bad = load_embeddings(hparams.z_dir+'test_11.txt')
    '''
    
    while i < 1:
        model2.sample_graph_slerp(hparams, placeholders, i,sample_1, sample_2, "slerp", (i+1)*0.1, num=70)
        model2.sample_graph_slerp(hparams, placeholders, i,sample_1, sample_2, "lerp", (i+1)*0.1, num=70)
        i+=1
    '''
    #'''
    while i < 100:
        model2.sample_graph(hparams, placeholders, i+hparams.offset, hparams.nodes, hparams.edges)
        i += 1
        #break
    #'''
    #G_good = load_
    #i = 0
    #while i < 10:
    #    model2.sample_graph(hparams, placeholders, i, hapams.nodes)
    #    i += 1

    #model2.plot_hspace(hparams, placeholders, 32)    
    #'''
