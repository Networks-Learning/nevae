from utils import *
from config import SAVE_DIR, VAEGConfig
from datetime import datetime
#from ops import print_vars
from cell import VAEGCell

import tensorflow as tf
import numpy as np
import logging
import pickle
import os
import random

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VAEG(VAEGConfig):
    def __init__(self, hparams, placeholders, num_nodes, num_features, edges, istest=False):
        self.features_dim = num_features
        self.input_dim = num_nodes
        self.dropout = placeholders['dropout']
        self.k = hparams.random_walk
        self.lr = placeholders['lr']
        self.decay = placeholders['decay']
        self.n = num_nodes
        self.d = num_features
        self.edges = edges
        #self.edges, self.non_edges = edges, non_edges


        #logger.info("Building model starts...")
        def neg_loglikelihood(prob_dict):
            '''
            negative loglikelihood of the edges
            '''
            ll = 0
            with tf.variable_scope('NLL'):
                dec_mat = tf.exp(tf.minimum(tf.reshape(prob_dict, [self.n, self.n]),tf.fill([self.n, self.n], 10.0)))
                dec_mat = tf.Print(dec_mat, [dec_mat], message="my decscore values:")
                print "Debug dec_mat", dec_mat.shape, dec_mat.dtype, dec_mat

		comp = tf.subtract(tf.ones([self.n, self.n], tf.float32), self.adj)
                comp = tf.Print(comp, [comp], message="my comp values:")

		temp = tf.reduce_sum(tf.multiply(comp,dec_mat))
		negscore = tf.fill([self.n,self.n], temp+1e-9)
                negscore = tf.Print(negscore, [negscore], message="my negscore values:")

		posscore = tf.multiply(self.adj, dec_mat)
                posscore = tf.Print(posscore, [posscore], message="my posscore values:")

                #dec_out = tf.multiply(self.adj, dec_mat) 
		softmax_out = tf.truediv(posscore, tf.add(posscore, negscore))
                ll = tf.reduce_sum(tf.log(tf.add(tf.multiply(self.adj, softmax_out), tf.fill([self.n,self.n], 1e-9))),1)
            return (-ll)

        def kl_gaussian(mu_1, sigma_1,debug_sigma, mu_2, sigma_2):
            '''
                Kullback leibler divergence for two gaussian distributions
            '''
            print sigma_1.shape, sigma_2.shape
            with tf.variable_scope("kl_gaussisan"):
                temp_stack = []
                for i in range(self.n):
                    temp_stack.append(tf.square(sigma_1[i]))
                first_term = tf.trace(tf.stack(temp_stack))

                temp_stack = []
                for i in range(self.n):
                    temp_stack.append(tf.matmul(tf.transpose( mu_1[i]),  mu_1[i]))
                second_term = tf.reshape(tf.stack(temp_stack), [self.n])

                #k = tf.fill([self.n], tf.cast(self.d, tf.float32))
                k = tf.fill([self.n], tf.cast(5, tf.float32))


                temp_stack = []
                #for i in range(self.n):
                #    temp_stack.append(tf.log(tf.truediv(tf.matrix_determinant(sigma_2[i]),tf.add(tf.matrix_determinant(sigma_1[i]), tf.fill([self.d, self.d], 1e-9)))))

                for i in range(self.n):
                    temp_stack.append(tf.reduce_prod(tf.square(debug_sigma[i])))

                print "Debug", tf.stack(temp_stack).shape
                third_term = tf.log(tf.add(tf.stack(temp_stack),tf.fill([self.n],1e-09)))

                print "debug KL", first_term.shape, second_term.shape, k.shape, third_term.shape, sigma_1[0].shape
                #return 0.5 *tf.reduce_sum((
                return 0.5 * tf.add(tf.subtract(tf.add(first_term ,second_term), k), third_term)

        def get_lossfunc(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma, dec_out):
                kl_loss = kl_gaussian(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma)  # KL_divergence loss
                likelihood_loss = neg_loglikelihood(dec_out)  # Cross entropy loss
                self.ll = likelihood_loss
                return tf.reduce_mean(kl_loss + likelihood_loss)


        self.adj = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='adj')
        self.features = tf.placeholder(dtype=tf.float32, shape=[self.n, self.d], name='features')
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.k, self.n, self.d], name='input')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[self.n, 5, 1], name='eps')

        self.cell = VAEGCell(self.adj, self.features)
        self.c_x, enc_mu, enc_sigma, debug_sigma,dec_out, prior_mu, prior_sigma, z_encoded = self.cell.call(self.input_data, self.n, self.d, self.k, self.eps, hparams.sample)
        self.prob = dec_out
        self.z_encoded = z_encoded
        self.cost = get_lossfunc(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma, dec_out)

        print_vars("trainable_variables")
        # self.lr = tf.Variable(self.lr, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.grad = self.train_op.compute_gradients(self.cost)
        self.grad_placeholder = [(tf.placeholder("float", shape=gr[1].get_shape()), gr[1]) for gr in self.grad]
        self.apply_transform_op = self.train_op.apply_gradients(self.grad)
        self.sess = tf.Session()

    def initialize(self):
        logger.info("Initialization of parameters")
        #self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

    def restore(self, savedir):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(savedir)
        print("Load the model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def next_batch(self):
        if self.current >= len(self.edges):
            return False
        adj = np.zeros([self.n, self.n])
        for (u,v) in self.edges[: self.current + self.batch_size]:
        #for (u,v) in edge_list:
            adj[u][v] = 1
            adj[v][u] = 1
        bias_matrix = basis(self.adj)
        self.current = self.current + self.batch_size
        feature = calculate_feature(self.adj)
        return (adj, bias_matrix, feature)

    def train(self,placeholders, hparams, adj, features):
        savedir = hparams.out_dir
        lr = hparams.learning_rate
        dr = hparams.dropout_rate
        decay = hparams.decay_rate

        # training
        num_epochs = hparams.num_epochs
        create_dir(savedir)
        ckpt = tf.train.get_checkpoint_state(savedir)
        saver = tf.train.Saver(tf.global_variables())

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)

        iteration = 1
        #1000
        for epoch in range(num_epochs):
            for batch in range(self.n_batches):
            #for i in range(len(adj)):
                adj, basis, features = self.next_batch()
                # Learning rate decay
                self.sess.run(tf.assign(self.lr, self.lr * (self.decay ** epoch)))

                feed_dict = construct_feed_dict(lr, dr, self.k, self.n, self.d, decay, placeholders)

                # sampled random from standard normal distribution
                eps = np.random.randn(self.n, self.n_z, 1)
                feed_dict.update({self.eps: eps})

                # The graph properties of the graph
                feed_dict.update({self.adj: adj})
                feed_dict.update({self.features: features})
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.basis: basis})

                grad_vals = self.sess.run([g[0] for g in self.grad], feed_dict=feed_dict)
                for j in xrange(len(self.grad_placeholder)):
                    feed_dict.update({self.grad_placeholder[j][0]: grad_vals[j]})
                input_, train_loss, _, probdict, cx= self.sess.run([self.input_data ,self.cost, self.apply_transform_op, self.prob, self.c_x], feed_dict=feed_dict)

                iteration += 1
                #print "Debug Grad", grad_vals[0]
                #print "Debug CX", cx
                if iteration % hparams.log_every == 0 and iteration > 0:
                    print("{}/{}(epoch {}), train_loss = {:.6f}".format(iteration, num_epochs, epoch + 1, train_loss))
		            #print(probdict)
                    checkpoint_path = os.path.join(savedir, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path, global_step=iteration)
                    logger.info("model saved to {}".format(checkpoint_path))




    def plot_hspace(self, hparams, placeholders, num):
            #plot the coordinate in hspace
            
            adj, deg = load_data(hparams.graph_file, num)
            hparams.sample= False
            for i in range(len(adj)):
                eps = np.random.randn(self.n, 5, 1) 
                feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
                feed_dict.update({self.adj: adj[i]})
	        feed_dict.update({self.features: deg[i]})
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.eps: eps})
                prob, ll, z = self.sess.run([self.prob, self.ll, self.z_encoded],feed_dict=feed_dict )
                with open(hparams.z_dir+'train'+str(i)+'.txt' , 'a') as f:
                    for z_i in z:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')
            
            #hparams.sample = True
            adj, deg = load_data(hparams.sample_file, num)
            for i in range(len(adj)):
                eps = np.random.randn(self.n, 5, 1) 
                feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
                feed_dict.update({self.adj: adj[i]})
	        feed_dict.update({self.features: deg[i]})
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.eps: eps})
                prob, ll, z = self.sess.run([self.prob, self.ll, self.z_encoded],feed_dict=feed_dict )
                with open(hparams.z_dir+'test_'+str(i)+'.txt', 'a') as f:
                    for z_i in z:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

    def sample_graph(self, hparams, placeholders, s_num, num=10, outdir=None):
        '''
        Args :
            num - int
                10
                number of edges to be sampled
            outdir - string
                output dir
            
        '''
        list_edges = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                    list_edges.append((i,j))
        
        candidate_edges =[ list_edges[i] for i in random.sample(range(len(list_edges)), num)]
        adj = np.zeros([self.n, self.n])
        for (u,v) in candidate_edges:
            adj[u][v] = 1
            adj[v][u] = 1
        
        deg = np.zeros([self.n, 1], dtype=np.float)

        for i in range(self.n):
            deg[i][0] = 2 * np.sum(adj[i])/(self.n*(self.n - 1))

        eps = np.random.randn(self.n, 5, 1) 
        with open(hparams.z_dir+'test_prior_'+str(s_num)+'.txt', 'a') as f:
                    for z_i in eps:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

        #tf.random_normal((self.n, 5, 1), 0.0, 1.0, dtype=tf.float32)
        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj})
	    feed_dict.update({self.features: deg})
        feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
        feed_dict.update({self.eps: eps})
        prob, ll = self.sess.run([self.prob, self.ll],feed_dict=feed_dict )
        
        prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
        prob = np.divide(prob, np.sum(prob))

        problist  = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                problist.append(prob[i][j])
        p = np.array(problist)
        p /= p.sum()
        
        candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[num], p=p, replace=False)]

        for (u,v) in candidate_edges:
            with open(hparams.sample_file+str(s_num)+'.txt', 'a') as f:
                        f.write(str(u)+' '+str(v)+' {}'+'\n')

        ll1 = np.mean(ll)
    
        with open(hparams.sample_file+'/ll.txt', 'a') as f:
            f.write(str(ll1)+'\n')
