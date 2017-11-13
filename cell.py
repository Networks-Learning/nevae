from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


#class VAEGCell(tf.nn.rnn_cell.RNNCell):
class VAEGCell(object):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, features):
        '''
        Args:
            x_dim - chunk_samples
            h_dim - rnn_size
            z_dim - latent_size
        '''
        self.adj = adj
        self.features = features
        #self.edges = edges
        #self.non_edges = non_edges
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
            c_x = input_layer(c_x, self.adj, self.features, k, n, d, activation=None, batch_norm=False, istrain=False, scope=None)
            #print "c_x",c_x.shape
	    with tf.variable_scope("Prior"):
                #prior_mu = tf.get_variable(name="prior_mu", shape=[n,d,1], initializer=tf.zeros_initializer())
                #prior_sigma = tf.matrix_diag(tf.ones(shape=[n,d]),name="prior_sigma")

                prior_mu = tf.get_variable(name="prior_mu", shape=[n,5,1], initializer=tf.zeros_initializer())
                prior_sigma = tf.matrix_diag(tf.ones(shape=[n,5]),name="prior_sigma")

	    print "Shape prior mu prior sigma", prior_mu.shape, prior_sigma.shape

	    with tf.variable_scope("Encoder"):
                list_cx = tf.unstack(c_x)
                # output will be of shape n X kd
                enc_hidden = fc_layer(tf.concat(list_cx,1), k*d, activation=tf.nn.relu, scope="hidden")
                #output will be of shape n X d
                
                '''
                enc_mu = fc_layer(enc_hidden, d, scope='mu')
		enc_mu = tf.reshape(enc_mu, [n,d,1])
                # output will be n X 1 then convert that to a diagonal matrix
                # enc_sigma = tf.matrix_diag(tf.transpose(fc_layer(enc_hidden, d, activation=tf.nn.softplus, scope='sigma'), name="enc_sigma"))
	        enc_sigma = tf.matrix_diag(tf.transpose(fc_layer(enc_hidden, d, activation=tf.nn.relu, scope='sigma'), name="enc_sigma"))
                '''

                enc_mu = fc_layer(enc_hidden, 5, scope='mu')
		enc_mu = tf.reshape(enc_mu, [n,5,1])
                # output will be n X 1 then convert that to a diagonal matrix
                # enc_sigma = tf.matrix_diag(tf.transpose(fc_layer(enc_hidden, d, activation=tf.nn.softplus, scope='sigma'), name="enc_sigma"))
	        enc_sigma = tf.matrix_diag(fc_layer(enc_hidden, 5, activation=tf.nn.relu, scope='sigma'), name="enc_sigma")

	    print "Shape encoder mu, sigma", enc_mu.shape, enc_sigma.shape

            # Random sampling ~ N(0, 1)
            #eps = tf.random_normal((n, d, 1), 0.0, 1.0, dtype=tf.float32)
            eps = tf.random_normal((n, 5, 1), 0.0, 1.0, dtype=tf.float32)

	    temp_stack = []
	    for i in range(n):
		temp_stack.append(tf.matmul(enc_sigma[i], eps[i]))
	    z = tf.add(enc_mu, tf.stack(temp_stack))
	    print "Shape z", z.shape
	    with tf.variable_scope("Decoder"):

	    	sum_negclass = 0.0
                z_stack = []
                for u in range(n):
                    for v in range(n):
			#print "Debug Shape",tf.concat(values=([z[u]], [z[v]]), axis = 1).shape 
                        z_stack.append(tf.concat(values=(tf.transpose(z[u]), tf.transpose(z[v])), axis = 1)[0])
	        
		print "Debug dec",tf.stack(z_stack).shape         
		dec_hidden = fc_layer(tf.stack(z_stack), 1, activation=tf.nn.softplus, scope = "hidden")

		#dec_mat = tf.exp(tf.reshape(dec_hidden, [n,n]))
                #print "Debug dec_mat", dec_mat.shape, dec_mat.dtype, dec_mat
		#comp = tf.subtract(tf.ones([n, n], tf.float32), self.adj)
		#temp = tf.reduce_sum(tf.multiply(comp,dec_mat))
		#negscore = tf.fill([n,n], temp+1e-9)
		#posscore = tf.multiply(self.adj, dec_mat)
		#dec_out = tf.multiply(self.adj, dec_mat) 
		#dec_out = tf.truediv(posscore, tf.add(posscore, negscore))
	print "shapes mu sig dec_out prior mu prio sig", enc_mu.shape, enc_sigma.shape, tf.convert_to_tensor(dec_hidden).shape, prior_mu.shape, prior_sigma.shape
        return (enc_mu, enc_sigma, dec_hidden, prior_mu, prior_sigma)

    def call(self,inputs,n,d,k):
        #with tf.variable_scope(self.name):
            return self.__call__(inputs,n,d,k)
