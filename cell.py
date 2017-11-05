from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp

class VAEGCell(tf.nn.rnn_cell.RNNCell):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, features, edges, non_edges, x_dim, h_dim, z_dim = 100):
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
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.n_h, state_is_tuple=True)

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, input, state,scope=None):
        '''
		Args:
			x - input 2D tensor [batch_size x 2*self.chunk_samples]
			state - tuple
				(hidden, cell_state)
			scope - string
				defaults to be None
    	'''
        adj,feature ,k, i = input
        n = get_shape(self.adj)[0]
        d = get_shape(self.feature)[1]
        with tf.variable_scope(scope or type(self).__name__):
            c_x = input_layer(adj, feature, k, i, activation=None, batch_norm=False, istrain=False, scope=None)
            #h, c = state
            with tf.variable_scope("Prior"):
                #prior_hidden = fc_layer(h, self.n_prior_hidden, activation = tf.nn.relu, scope = "hidden")
                prior_mu = tf.get_variable(name="prior_mu", shape=[n,d], initializer=tf.zeros_initializer())
                #tf.get_variable(name="prior_mu", shape=[n,self.n_x,1], initialiser=) #fc_layer(prior_hidden, self.n_z, scope = "mu")
                prior_sigma = tf.diag(np.ones(shape=[1,n]),name="prior_sigma") #fc_layer(prior_hidden, self.n_z, activation = tf.nn.softplus, scope = "sigma")# >=0

            cx_1 = fc_layer(tf.matmul(c_x, get_basis(adj)), [n,d], scope="phi_C")# >=0

            with tf.variable_scope("Encoder"):
                enc_hidden = fc_layer(tf.concat(values=(cx_1, c_x[i]), axis=1), self.n_enc_hidden, activation = tf.nn.relu, scope = "hidden")
                enc_mu = fc_layer(enc_hidden, [n,d], scope = 'mu', name="enc_mu")
                enc_sigma = tf.diag(fc_layer(enc_hidden, [1,n], activation = tf.nn.softplus, scope = 'sigma'), name="enc_sigma")

            # Random sampling ~ N(0, 1)
            eps = tf.random_normal((n, d), 0.0, 1.0, dtype=tf.float32)
            # z = mu + sigma*epsilon, latent variable from reparametrization trick
            z = tf.add(enc_mu, tf.multiply(enc_sigma, eps))
            #z_1 = fc_layer(z, self.n_z_1, activation = tf.nn.relu, scope = "phi_z")

            with tf.variable_scope("Decoder"):
                sum_negclass = 0
                dec_out = tf.get_variable(name="dec_out", shape=[n,n]) #fc_layer(prior_hidden, self.n_z, scope = "mu")
                #dec_in = tf.get_variable(name="dec_out",
                #                          shape=[self.n_x, 1])  # fc_layer(prior_hidden, self.n_z, scope = "mu")
                # for i in range(n):
                #     for j in range(n):
                #         dec_in[i][j] = tf.concat(values=(z[i], z[j]))
                # dec_out = fc_layer(tf.concat(dec_in, self.n_dec_hidden, activation = tf.nn.relu, scope = "hidden"))
                for (u,v) in self.non_edges:
                     sum_negclass += exp(fc_layer(tf.concat(values=(z[u], z[v]), axis=1), self.n_dec_hidden, activation = tf.nn.relu, scope = "hidden"))
                for (u,v) in self.edges:
                     dec = fc_layer(tf.concat(values=(z[u], z[v]), axis=1), self.n_dec_hidden, activation = tf.nn.relu, scope = "hidden")
                     dec_out[u][v] = exp(dec) / (exp(dec) + sum_negclass)
                #dec_sigma = fc_layer(dec_hidden, self.n_x, activation = tf.nn.softplus, scope = "sigma")

            #output, next_state = self.lstm(, state)

        return (enc_mu, enc_sigma, dec_out, prior_mu, prior_sigma)
