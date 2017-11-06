import tensorflow as tf
from utils import *

def input_layer(c_mat, adj, feature, k,n,d,activation = None, batch_norm = False, istrain = False, scope = None):
    w_in = tf.get_variable(name="w_in", shape=[k,d, d], initializer=tf.contrib.layers.xavier_initializer())
    #c_mat = tf.get_variable(name="C", shape=[k, n, d], initializer=tf.contrib.layers.xavier_initializer())
    output_list = []
    for i in range(k):
        if i > 0:
            #c_mat[i] = tf.add(tf.transpose(tf.matmul(w_in[i], tf.transpose(feature))),tf.matmul(adj, c_mat[i-1]))
	    output_list.append( tf.add(tf.transpose(tf.matmul(w_in[i], tf.transpose(feature))),tf.matmul(adj, output_list[i-1])))
        else:
            #c_mat[i] 
	    output_list.append(tf.transpose(tf.matmul(w_in[i], tf.transpose(feature))))
    
    return tf.stack(output_list)



def fc_layer(input_, output_size, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    fully convlolution layer
    Args :
        input_  - 2D tensor
            general shape : [batch, input_size]
        output_size - list of the sizes for the output
            shape of output 2D tensor
        activation - activation function
            defaults to be None
        batch_norm - bool
            defaults to be False
            if batch_norm to apply batch_normalization
        istrain - bool
            defaults to be False
            indicator for phase train or not
        scope - string
            defaults to be None then scope becomes "fc"
    '''
    with tf.variable_scope(scope or "fc"):
        w = tf.get_variable(name="w", shape = [get_shape(input_)[1]]+ output_size, initializer=tf.contrib.layers.xavier_initializer())
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(tf.matmul(input_, w) , center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = output_size, initializer=tf.constant_initializer(0.01))
            if activation is None:
                return tf.nn.xw_plus_b(input_, w, b)
            return activation(tf.nn.xw_plus_b(input_, w, b))
