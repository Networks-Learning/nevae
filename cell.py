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


    def __call__(self,c_x,n,d,k, scope=None):
        '''
		Args:
			x - input 2D tensor [batch_size x 2*self.chunk_samples]
			state - tuple
				(hidden, cell_state)
			scope - string
				defaults to be None
    	'''
        #adj,feature ,k, i = input
        #n = get_shape(self.adj)[0]
        #d = get_shape(self.features)[1]
        with tf.variable_scope(scope or type(self).__name__):
            #print "c_x shape",c_x.shape
            c_x = input_layer(c_x, self.adj, self.features, k, n, d, activation=None, batch_norm=False, istrain=False, scope=None)
            #print "c_x shape",c_x.shape
	    with tf.variable_scope("Prior"):
                prior_mu = tf.get_variable(name="prior_mu", shape=[n,d], initializer=tf.zeros_initializer())
                prior_sigma = tf.matrix_diag(tf.ones(shape=[1,n]),name="prior_sigma")[0]

	    with tf.variable_scope("Encoder"):
                list_cx = tf.unstack(c_x)
                # output will be of shape n X kd
                enc_hidden = fc_layer(tf.concat(list_cx,1), k*d, activation=tf.nn.relu, scope="hidden")
                #output will be of shape n X d
                enc_mu = fc_layer(enc_hidden, d, scope='mu')
                # output will be n X 1 then convert that to a diagonal matrix
                enc_sigma = tf.matrix_diag(tf.transpose(fc_layer(enc_hidden, 1, activation=tf.nn.softplus, scope='sigma'), name="enc_sigma"))[0]
            #with tf.variable_scope("Encoder"):
            #    enc_hidden = fc_layer(c_x, [k,d], activation=tf.nn.relu, scope="hidden")
            #    enc_mu = fc_layer(enc_hidden, [n,d], scope = 'mu', name="enc_mu")
            #    enc_sigma = tf.diag(fc_layer(enc_hidden, [1,n], activation = tf.nn.softplus, scope = 'sigma'), name="enc_sigma")

            # Random sampling ~ N(0, 1)
            eps = tf.random_normal((n, d), 0.0, 1.0, dtype=tf.float32)
            #print "Shape eps, nmu, sig", eps.shape, enc_mu.shape, enc_sigma.shape
	    z = tf.add(enc_mu, tf.multiply(enc_sigma, eps))
	    #print "Shape z", z[0].shape
	    with tf.variable_scope("Decoder"):

	    	sum_negclass = 0.0
                #dec_out = np.zeros([n,n])
                z_stack = []
                #tf.get_variable(name="dec_out", shape=[n,n])
                for u in range(n):
                    for v in range(n):
			#print "Debug Shape",tf.concat(values=([z[u]], [z[v]]), axis = 1).shape 
                        z_stack.append(tf.concat(values=([z[u]], [z[v]]), axis = 1)[0])
                # size is n^2 X 1
       		#print z_stack.shape
	        
		#print tf.stack(z_stack).shape         
		dec_hidden = fc_layer(tf.stack(z_stack), 1, activation = tf.nn.relu, scope = "hidden")
                #print "Debug dec_hidden", dec_hidden.shape, dec_hidden.dtype, dec_hidden

		dec_mat = tf.exp(tf.reshape(dec_hidden, [n,n]))
                print "Debug dec_mat", dec_mat.shape, dec_mat.dtype, dec_mat
		comp = tf.subtract(tf.ones([n, n], tf.float32), self.adj)
		temp = tf.reduce_sum(tf.matmul(comp,dec_mat))
		#print temp
		negscore = tf.fill([n,n], temp)
		posscore = tf.matmul(self.adj, dec_mat)
		dec_out = tf.truediv(posscore, tf.add(posscore, negscore))
	print "shapes mu sig dec_out prior mu prio sig", enc_mu.dtype, enc_sigma.dtype, tf.convert_to_tensor(dec_out).dtype, prior_mu.dtype, prior_sigma.dtype
        return (enc_mu, enc_sigma, tf.convert_to_tensor(dec_out), prior_mu, prior_sigma)

    def call(self,inputs,n,d,k):
        #with tf.variable_scope(self.name):
            return self.__call__(inputs,n,d,k)
