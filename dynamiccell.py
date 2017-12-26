from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


class VAEGDCell(tf.nn.rnn_cell.RNNCell):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, features, h_dim, x_dim, z_dim):
        '''
        Args:
        adj : adjacency matrix
        features: feature matrix

        hyperparameters
        h_dim : hidden state dimension
        x_dim : input chunk size (number of edges to be processed at the same time)
        z_dim : encoded space dimension
        '''

        self.adj = adj
        self.features = features
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim
        self.name = self.__class__.__name__.lower()
        self.lstm = tf.nn.rnn_cell.LSTM(self.n_h, state_is_tuple=True)

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, state, c_x, n, d, k, eps_passed, sample, bias_laplace, scope=None):
        '''
		Args:
			c_x - tensor to be filled up with random walk property
			n - number of nodes
			d - number of features in the feature matrix
			k - length of random walk
				defaults to be None
    	'''
        c, h = state
        with tf.variable_scope(scope or type(self).__name__):
            c_x = input_layer(c_x, self.adj, self.features, k, n, d, activation=None, batch_norm=False, istrain=False,
                              scope=None)

        with tf.variable_scope("Prior"):
            prior_hidden = fc_layer(h, self.n_prior_hidden, activation=tf.nn.relu, scope="hidden")
            prior_mu = fc_layer(prior_hidden, self.n_z, scope="mu")
            prior_sigma = fc_layer(prior_hidden, self.n_z, activation=tf.nn.softplus, scope="sigma")  # >=0

        c_x_1 = fc_layer(tf.matmul(c_x[-1],bias_laplace), self.n_x_1, activation = tf.nn.relu, scope = "phi_c")# >=0

        with tf.variable_scope("Encoder"):
                list_cx = tf.unstack(c_x)
                list_cx.append(c_x_1)
                list_cx.append(h)
                # output will be of shape n X kd
                enc_hidden = fc_layer(tf.concat(list_cx, 1), self.n_enc_hidden, activation=tf.nn.relu, scope="hidden")
                # output will be of shape n X 5 (this is a hyper paramater)
                enc_mu = fc_layer(enc_hidden, self.n_z, activation=tf.nn.softplus, scope='mu')
                enc_mu = tf.reshape(enc_mu, [n, self.n_z, 1])

                # output will be n X 1 then convert that to a diagonal matrix
                intermediate_sigma = fc_layer(enc_hidden, self.n_z, activation=tf.nn.softplus, scope='sigma')
                intermediate_sigma = tf.Print(intermediate_sigma, [intermediate_sigma], message="my debug_sigma-values:")
                enc_sigma = tf.matrix_diag(intermediate_sigma, name="enc_sigma")


        # Random sampling ~ N(0, 1)
        eps = eps_passed
        # tf.random_normal((n, 5, 1), 0.0, 1.0, dtype=tf.float32)

        temp_stack = []
        for i in range(n):
            temp_stack.append(tf.matmul(enc_sigma[i], eps[i]))
            z = tf.add(enc_mu, tf.stack(temp_stack))
            # While we are trying to sample some edges, we sample Z from prior
        if sample:
            z = eps

        with tf.variable_scope("Decoder"):
                z_stack = []
                for u in range(n):
                    # Assuming the graph is undirected
                    for v in range(u+1, n):
                        z_stack.append(tf.concat(values=(tf.transpose(z[u]), tf.transpose(z[v])), axis=1)[0])
                dec_hidden = fc_layer(tf.stack(z_stack), 1, activation=tf.nn.softplus, scope="hidden")

        return (c_x, enc_mu, enc_sigma, dec_hidden, prior_mu, prior_sigma, z)

    def call(self, state, c_x, n, d, k, eps_passed, sample, bias_laplace):
        return self.__call__(state, c_x, n, d, k, eps_passed, sample, bias_laplace)
