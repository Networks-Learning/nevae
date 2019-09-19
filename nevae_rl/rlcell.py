import tensorflow as tf
from layer import *
import numpy as np
from utils import *
from math import exp


class VAEGRLCell(object):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, features, z_dim, bin_dim, edges, enc_mu=None, enc_sigma=None):
        '''
        Args:
        adj : adjacency matrix
        features: feature matrix
        '''
        self.adj = adj
        self.features = features
        self.z_dim = z_dim
        #self.weight = weight
        self.name = self.__class__.__name__.lower()
        self.bin_dim = bin_dim
        self.edges = edges
        #self.enc_mu = enc_mu
        #self.enc_sigma = enc_sigma

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, c_x, n, d, k, combination, eps_passed, sample, scope=None):
        '''
		Args:
			c_x - tensor to be filled up with random walk property
			n - number of nodes
			d - number of features in the feature matrix
			k - length of random walk
				defaults to be None
    	'''
        with tf.variable_scope(scope or type(self).__name__):
            eps = eps_passed
            '''
            temp_stack = []
            print("Debug edges", tf.shape(self.edges))
            for i in range(n):
                temp_stack.append(tf.matmul(self.enc_sigma[i], eps[i]))
            z = tf.add(self.enc_mu, tf.stack(temp_stack))
            # While we are trying to sample some edges, we sample Z from prior
            if sample:
            '''
            z = eps
            z = tf.Print(z,[z], message="my z-values RL:")

            with tf.variable_scope("Poisson"):
                z_reshape = tf.reshape(z,[-1,self.z_dim])
                z_reshape = tf.Print(z_reshape,[z_reshape], message="my z-reshape-values:")
                n_cast=tf.fill([1, self.z_dim], tf.cast(n, dtype=tf.float32))
                z_concat = tf.concat(values = (z_reshape, n_cast), axis = 0)
                #print("Debug concat z", z_concat.get_shape())
                lambda_edge = fc_layer(z_concat, 1, activation=tf.nn.softplus, scope = "edge")
            
            def loop_cond(t, k, z, z_stack, z_stack_weight):
                N = tf.stack([tf.shape(t)[0]])[0]
                #N= tf.Print(N, [N], message="my N-values")
                return tf.less(k, N)

            def body(t, k, z, z_stack, z_stack_weight):
                # need to check once sanity
                dots = tf.concat(values = ([tf.gather(z,t[k][0])], [tf.gather(z,t[k][1])]), axis = 1)
                for j in range(self.bin_dim):
                    m = np.zeros((1, self.bin_dim))
                    m[0][j] = 1
                    temp =  tf.concat(values = (dots, tf.cast(m, tf.float32)), axis=1)
                    z_stack_weight = tf.concat(values = (z_stack_weight, temp), axis = 0)
                return (t,k+1,z,tf.concat(values=(z_stack,dots), axis=0), z_stack_weight)
            
            k = tf.constant(0)
            z_new = tf.reshape(z, [n,self.z_dim])
            print("Debug z shape", z_new.get_shape())
            dec_hidden = []
            weight = []

            with tf.variable_scope("DecoderRL", reuse=tf.AUTO_REUSE):
                z_stack_label = []
                for u in range(n):
                    for j in range(4):
                        # we considered 4 types of atom C, H, O, N
                        m = np.zeros((1, 4))
                        m[0][j] = 1
                        z_stack_label.append(tf.concat(values=(tf.transpose(z[u]),m), axis = 1)[0])
                label = fc_layer(tf.stack(z_stack_label), 1, activation=tf.nn.softplus, scope = "label")

            for i in range(combination):
                z_stack = tf.constant(0, shape=[1, 2 * self.z_dim], dtype = tf.float32)
                z_stack_weight = tf.constant(0, shape=[1, 2 * self.z_dim+self.bin_dim], dtype = tf.float32)
                t = self.edges[i]
                _,_,_,z_stack,z_stack_weight = tf.while_loop(loop_cond, body, [t,k,z_new,z_stack, z_stack_weight], shape_invariants=[t.get_shape(), k.get_shape(), z_new.get_shape(), tf.TensorShape([None, 2 * self.z_dim]), tf.TensorShape([None, 2 * self.z_dim+self.bin_dim])])
                with tf.variable_scope("DecoderRL", reuse=tf.AUTO_REUSE):
                    dec_hidden.append(fc_layer(z_stack[1:], 1, activation=tf.nn.softplus, scope = "hidden"))
                    weight.append(fc_layer(z_stack_weight[1:], 1, activation=tf.nn.softplus, scope = "marker"))
                    #label.append(fc_layer(tf.stack(z_stack_label), 1, activation=tf.nn.relu, scope = "label"))
        return (dec_hidden, weight, tf.reduce_mean(lambda_edge), label)

    def call(self, inputs, n, d, k, combination, eps_passed, sample):
        return self.__call__(inputs, n, d, k, combination, eps_passed, sample)
