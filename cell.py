from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


#class VAEGCell(tf.nn.rnn_cell.RNNCell):
class VAEGCell(object):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, features, edges, non_edges):
        '''
        Args:
            x_dim - chunk_samples
            h_dim - rnn_size
            z_dim - latent_size
        '''
        self.adj = adj
        self.features = features
        self.edges = edges
        self.non_edges = non_edges
        self.name = self.__class__.__name__.lower()

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h


    def __call__(self,k, scope=None):
        '''
		Args:
			x - input 2D tensor [batch_size x 2*self.chunk_samples]
			state - tuple
				(hidden, cell_state)
			scope - string
				defaults to be None
    	'''
        #adj,feature ,k, i = input
        n = get_shape(self.adj)[0]
        d = get_shape(self.features)[1]
        with tf.variable_scope(scope or type(self).__name__):
            c_x = input_layer(self.adj, self.features, k, activation=None, batch_norm=False, istrain=False, scope=None)
            with tf.variable_scope("Prior"):
                prior_mu = tf.get_variable(name="prior_mu", shape=[n,d], initializer=tf.zeros_initializer())
                prior_sigma = tf.diag(np.ones(shape=[1,n]),name="prior_sigma")

            with tf.variable_scope("Encoder"):
                enc_hidden = fc_layer(c_x, self.n_enc_hidden, activation=tf.nn.relu, scope="hidden")
                enc_mu = fc_layer(enc_hidden, [n,d], scope = 'mu', name="enc_mu")
                enc_sigma = tf.diag(fc_layer(enc_hidden, [1,n], activation = tf.nn.softplus, scope = 'sigma'), name="enc_sigma")

            # Random sampling ~ N(0, 1)
            eps = tf.random_normal((n, d), 0.0, 1.0, dtype=tf.float32)
            z = tf.add(enc_mu, tf.multiply(enc_sigma, eps))

            with tf.variable_scope("Decoder"):
                sum_negclass = 0
                dec_out = tf.get_variable(name="dec_out", shape=[n,n])
                for (u,v) in self.non_edges:
                     sum_negclass += exp(fc_layer(tf.concat(values=(z[u], z[v]), axis=1), self.n_dec_hidden, activation = tf.nn.relu, scope = "hidden"))
                for (u,v) in self.edges:
                     dec = fc_layer(tf.concat(values=(z[u], z[v]), axis=1), self.n_dec_hidden, activation = tf.nn.relu, scope = "hidden")
                     dec_out[u][v] = exp(dec) / (exp(dec) + sum_negclass)

        return (enc_mu, enc_sigma, dec_out, prior_mu, prior_sigma)

    def call(self,k):
        #with tf.variable_scope(self.name):
            return self.__call__(k)
