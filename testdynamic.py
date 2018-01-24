import tensorflow as tf

data = tf.placeholder(tf.float32, [None, 20,1])
target = tf.placeholder(tf.float32, [None, 21])

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
print 

