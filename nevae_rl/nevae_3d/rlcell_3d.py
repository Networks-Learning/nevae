from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp

class VAEGRLCell(object):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, weight, features, coord, z_dim, bin_dim, node_count, edges):
        '''
        Args:
        adj : adjacency matrix
	features: feature matrix
	'''
        self.adj = adj
        self.features = features
        self.z_dim = z_dim
        #self.weight = weight
	self.coord = coord
        self.name = self.__class__.__name__.lower()
        self.bin_dim = bin_dim
        self.node_count = node_count
        self.edges = edges
    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, c_x, n, d, k, combination, z_coord, eps_passed, sample, scope=None):
        '''
		Args:
			c_x - tensor to be filled up with random walk property
			n - number of nodes 
			d - number of features in the feature matrix
			k - length of random walk
				defaults to be None
    	'''
        with tf.variable_scope(scope or type(self).__name__):
            # Random sampling ~ N(0, 1)
            eps = eps_passed 
            z = eps
            z = tf.Print(z,[z], message="my z-values:")

            with tf.variable_scope("DecoderRL", reuse=True):
                    z_stack_label = []
		    for u in range(n):
                        for j in range(4):
                            m = np.zeros((1, 4))
                            m[0][j] = 1
                            z_stack_label.append(tf.concat(values=(tf.transpose(z[u]),m), axis = 1)[0])
                    label = fc_layer(tf.stack(z_stack_label), 1, activation=tf.nn.softplus, scope = "label")
		    coor_mu = fc_layer(tf.stack(z_coord), 3, activation=tf.nn.softplus, scope='coor_mu')    
                    coor_sigma = fc_layer(tf.stack(z_coord), 6, activation=tf.nn.softplus, scope='coor_sigma')
		    coor_sigma_list = [] 
            	    for u in range(n):
		    	temp_sigma = tf.contrib.distributions.fill_triangular(coor_sigma[u], upper=True)
                        diag_mat = tf.matrix_diag(tf.cast([9, 9, 9], tf.float32))
                	coor_sigma_list.append(tf.add(0.5 * (temp_sigma + tf.transpose(temp_sigma)),diag_mat) )
	    

        return (coor_mu, coor_sigma_list)

    def call(self,inputs, n, d, k, combination, z_coord, eps_passed, sample):
            return self.__call__(inputs, n, d, k, combination, z_coord, eps_passed, sample)
