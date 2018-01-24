from utils import *
from config import SAVE_DIR, VAEGConfig
from datetime import datetime
#from ops import print_vars
from cell import VAEGCell
from dynamiccell import VAEGDCell

from math import ceil, log
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
        self.z_dim = hparams.z_dim
        self.x_dim = hparams.z_dim
        self.h_dim = num_nodes
        self.n_seq = hparams.n_seq
        self.seq_size = 0
        self.current = 0
        self.bin_dim = hparams.bin_dim
        #self.edges, self.non_edges = edges, non_edges

        #logger.info("Building model starts...")
        def neg_loglikelihood(prob_dict, w_edge):
            '''
            negative loglikelihood of the edges
            '''
            ll = 0
            with tf.variable_scope('NLL'):
                dec_mat = tf.exp(tf.minimum(tf.reshape(prob_dict, [self.n, self.n]),tf.fill([self.n, self.n], 10.0)))
                dec_mat = tf.Print(dec_mat, [dec_mat], message="my decscore values:")
                print "Debug dec_mat", dec_mat.shape, dec_mat.dtype, dec_mat
                w_edge_new = tf.reshape(w_edge, [self.n, self.n, self.bin_dim])
            print("DEBUG weight bin", self.weight_bin.shape)
            weight_temp = tf.multiply(self.weight_bin, w_edge_new)
            weight_stack = []
            weight_negative = []
            #np.zeros([self.n, self.n])
            for i in range(self.n):
                for j in range(self.n):
                    weight_stack.append(tf.reduce_sum(weight_temp[i][j]))
                    weight_negative.append(tf.reduce_sum(w_edge_new[i][j]))
            weight_stack = tf.reshape(weight_stack, [self.n, self.n])
            weight_negative = tf.reshape(weight_negative, [self.n, self.n])
            w_score = tf.truediv(weight_stack, weight_negative)
            
            comp = tf.subtract(tf.ones([self.n, self.n], tf.float32), self.adj)
            comp = tf.Print(comp, [comp], message="my comp values:")

            #temp = tf.reduce_sum(tf.multiply(comp,dec_mat))
            temp = tf.reduce_sum(tf.multiply(tf.multiply(comp,dec_mat), w_score))
            negscore = tf.fill([self.n,self.n], temp+1e-9)
            negscore = tf.Print(negscore, [negscore], message="my negscore values:")

            posscore = tf.multiply(self.adj, dec_mat)
            posscore = tf.Print(posscore, [posscore], message="my posscore values:")

            posweightscore = tf.multiply(posscore, w_score)
            posweightscore = tf.Print(posweightscore, [posweightscore], message="my weighted posscore")
            #dec_out = tf.multiply(self.adj, dec_mat)
            softmax_out = tf.truediv(posscore, tf.add(posweightscore, negscore))
            ll = tf.reduce_sum(tf.log(tf.add(tf.multiply(self.adj, softmax_out), tf.fill([self.n,self.n], 1e-9))),1)
            #ll = tf.Print(ll, [ll], message="my loss values:")
            return (-ll)

        def kl_gaussian(mu_1, sigma_1,debug_sigma_1, debug_sigma_2, mu_2, sigma_2):
            '''
                Kullback leibler divergence for two gaussian distributions
            '''
            #print("Debug sigma1", debug_sigma_1, len(debug_sigma_1[0]))
            #print sigma_1.shape, sigma_2.shape
            with tf.variable_scope("kl_gaussian"):
                temp_stack_1 = []
                temp_stack_2 = []
                for i in range(self.n):
                    #print("DEBUG i", i)
                    temp_stack_1.append(tf.reduce_prod(debug_sigma_1[i]))
                    temp_stack_2.append(tf.reduce_prod(debug_sigma_2[i]))

                # Inverse of diaginal covariance
                inverse_sigma_2 = tf.matrix_diag(tf.truediv(tf.ones(tf.shape(debug_sigma_2)), debug_sigma_2))

                term_2 = []
                for i in range(self.n):
                    term_2.append(tf.matmul(inverse_sigma_2[i], sigma_1[i]))
                # Difference between the mean
                term_3 = []
                k = tf.fill([self.n], tf.cast(tf.shape(mu_1)[1], tf.float32))
                diff_mean = tf.subtract(mu_2, mu_1)
                
                for i in range(self.n):
                    term_3.append(tf.matmul(tf.matmul(tf.transpose(diff_mean[i]), inverse_sigma_2[i]), diff_mean[i]))
                term_3 = tf.Print(term_3, [term_3], message="my term_3 values:")
                KL = (0.5 *
                        (tf.log(tf.truediv(temp_stack_2, temp_stack_1)) 
                         + tf.trace(term_2) 
                         + term_3
                         - k))
                KL = tf.Print(KL, [KL], message="my KL values:")

                #print("Debug mu1", tf.shape(mu_1)[1])
                return tf.reduce_sum(KL)

        def get_lossfunc(enc_mu, enc_sigma, enc_debug_sigma,prior_mu, prior_sigma, prior_debug_sigma, dec_out, w_edge):
                kl_loss = kl_gaussian(enc_mu, enc_sigma, enc_debug_sigma,prior_debug_sigma,prior_mu, prior_sigma)  # KL_divergence loss
                likelihood_loss = neg_loglikelihood(dec_out, w_edge)  # Cross entropy loss
                self.ll = likelihood_loss
            
                return tf.reduce_mean(kl_loss + likelihood_loss)

        self.adj = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='adj')
        self.features = tf.placeholder(dtype=tf.float32, shape=[self.n, self.d], name='features')
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.k, self.n, self.d], name='input')
        self.weight = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='weight')
        self.weight_bin = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n, self.bin_dim], name='weight_bin')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[self.n, self.z_dim, 1], name='eps')
        self.basis = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='basis')

        # Based on the static or dynamic case this is done
        if not hparams.dynamic:
            self.cell = VAEGCell(self.adj, self.features)
        else:
            print("Debug Dynamic")
            #sel.batch_size = self.n
            self.cell = VAEGDCell(self.adj, self.features, self.basis, hparams.sample, self.eps, self.k, self.h_dim, self.x_dim, self.z_dim, self.bin_dim)
            self.initial_state_c, self.initial_state_h = self.cell.zero_state(batch_size=1, dtype=tf.float32)
        
        if not hparams.dynamic:
            self.c_x, enc_mu, enc_sigma, debug_sigma, dec_out, prior_mu, prior_sigma, z_encoded =\
                self.cell.call(self.input_data, self.n, self.d, self.k, self.eps, hparams.sample)
            self.cost = get_lossfunc(enc_mu, enc_sigma, debug_sigma, prior_mu, prior_sigma, debug_sigma, dec_out)
        else:
            #c_x, enc_mu, enc_sigma, enc_intermediate_sigma, dec_hidden, prior_mu, prior_sigma, prior_intermediate_sigma, z
            #(self.c_x, enc_mu, enc_sigma, debug_sigma1, dec_out, prior_mu, prior_sigma, debug_sigma2, z_encoded), last_state = \
            #self.c_x, enc_mu, enc_sigma, debug_sigma1,dec_out, prior_mu, prior_sigma, debug_sigma2, z_encoded = a
            outputs, last_state = tf.contrib.rnn.static_rnn(self.cell, [self.input_data], initial_state=(self.initial_state_c, self.initial_state_h))
            #print("Debug a", a) 
            #self.c_x, enc_mu, enc_sigma, debug_sigma1,dec_out, prior_mu, prior_sigma, debug_sigma2, z_encoded = a

            #names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "dec_rho", "prior_mu", "prior_sigma"]
            names = ["c_x", "enc_mu", "enc_sigma", "enc_intermediate_sigma", "dec_hidden", "prior_mu", "prior_sigma", "prior_intermediate_sigma", "z", "weight"]
            outputs_decoupled = []
            
            for n,name in enumerate(names):
                with tf.variable_scope(name):
                    for o in outputs:
                        x =o[n]
                    #x = [o[n] for o in a]
                        outputs_decoupled.append(x)
            print("Debug 1", outputs_decoupled, len(outputs_decoupled))
            self.c_x, enc_mu, enc_sigma, debug_sigma1,dec_out, prior_mu, prior_sigma, debug_sigma2, z_encoded, self.w_edge = outputs_decoupled
            self.final_state_c,self.final_state_h = last_state
            #print("Debug 2", debug_sigma1, debug_sigma2.shape)
            self.cost = get_lossfunc(enc_mu, enc_sigma, debug_sigma1, prior_mu, prior_sigma, debug_sigma2, dec_out, self.w_edge)
        self.prob = dec_out
        self.z_encoded = z_encoded
        #self.cost = get_lossfunc(enc_mu, enc_sigma, debug_sigma1,prior_mu, prior_sigma, debug_sigma2, dec_out)

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

    def next_seq(self, i):
        if self.current >= len(self.edges[i]):
            return (True, -1, -1, -1)
        weight_mat = np.zeros([self.n, self.n])
        #print("Debug ", len(self.edges[i]), self.seq_size)
        for (u,v) in self.edges[i][: self.current + self.seq_size]:
        #for (u,v) in edge_list:
            weight_mat[u][v] += 1
            weight_mat[v][u] += 1
        #bias_matrix = basis(adj)
        self.current = self.current + self.seq_size
        feature, weight_bin, adj = calculate_feature(weight_mat, self.bin_dim)
        bias_matrix = basis(adj)
        #print()
        return (False, adj, bias_matrix, feature, weight_bin, weight_mat)

    def train(self,placeholders, hparams, adj, features, edges):
        savedir = hparams.out_dir
        lr = hparams.learning_rate
        dr = hparams.dropout_rate
        decay = hparams.decay_rate

        #training
        num_epochs = hparams.num_epochs
        create_dir(savedir)
        ckpt = tf.train.get_checkpoint_state(savedir)
        saver = tf.train.Saver(tf.global_variables())

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)
        
        iteration = 1

        for epoch in range(num_epochs):
            for i in range(len(adj)):
                # sequence size and batch size is 1
                final_state_c, final_state_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))
                #self.cell.zero_state(batch_size=1, dtype=tf.float32)
                print("DEBUG LEN",len(self.edges[i]) // self.n_seq)
                self.current = 0
                self.seq_size = int(ceil(len(self.edges[i]) // self.n_seq))
                for seq in range(self.n_seq):
                    #print('batch', batch)
                    #for i in range(len(adj)):
                    #specific to dynamic one
                    finished, adjnew, basis, features, weight_bin, weight = self.next_seq(i)
                    print("DEBUG RAW DATA", weight_bin.shape) 
                    # Learning rate decay
                    #self.sess.run(tf.assign(self.lr, self.lr * (self.decay ** epoch)))

                    feed_dict = construct_feed_dict(lr, dr, self.k, self.n, self.d, decay, placeholders)

                    # sampled random from standard normal distribution
                    eps = np.random.randn(self.n, self.z_dim, 1)
                    feed_dict.update({self.eps: eps})
                    
                    # The graph properties of the graph
                    feed_dict.update({self.adj: adjnew})
                    feed_dict.update({self.features: features})
                    feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                    feed_dict.update({self.basis: basis})
                    feed_dict.update({self.weight_bin: weight_bin})
                    feed_dict.update({self.weight: weight})
                    feed_dict.update({self.initial_state_c: final_state_c})
                    feed_dict.update({self.initial_state_h: final_state_h})
                    #self.initial_state_c, self.initial_state_h = self.cell.zero_state(batch_size=1, dtype=tf.float32)

                    '''
                    print("Debug grad", self.grad)
                    for g in self.grad:
                        print("DEBUG", g[0])
                        self.sess.run(g[0], feed_dict=feed_dict)
                    grad_vals = self.sess.run([g[0] for g in self.grad], feed_dict=feed_dict)
                    for j in xrange(len(self.grad_placeholder)):
                        feed_dict.update({self.grad_placeholder[j][0]: grad_vals[j]})
                    '''
                    input_, train_loss, _, probdict, cx, final_state_c, final_state_h = self.sess.run([self.input_data ,self.cost, self.apply_transform_op, self.prob, self.c_x, self.final_state_c, self.final_state_h], feed_dict=feed_dict)

                    iteration += 1
                    if iteration % hparams.log_every == 0 and iteration > 0:
                        print("{}/{}(epoch {}), train_loss = {:.6f}".format(iteration, epoch*len(adj)*seq, num_epochs*len(adj)*self.n_seq, train_loss))
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
                #print('LL', ll)
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

    
    
    def get_stat(self, hparams, edges, placeholders):
        
        num_edges = hparams.sample_edge
        #edges_seq = 1
        edges_seq = int(ceil(num_edges // self.n_seq))
        k = 0
        final_state_c, final_state_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))
        feature = np.zeros([self.n, 1], dtype=np.float)
        adj = np.zeros([self.n, self.n])
        probtotal = 0.0
        print("Debug edges", edges, edges_seq)
        
        while k < num_edges:
            #if k == 0:
            #    candidate_edges = edges[:1]
            #else:
            candidate_edges = edges[k:k+edges_seq]
            for (u,v) in candidate_edges:
                adj[u][v] = 1
                adj[v][u] = 1
        
            for i in range(self.n):
                feature[i][0] =  np.sum(adj[i])//(self.n - 1)
            
            basis_matrix = basis(adj)

            eps = np.random.randn(self.n, self.z_dim, 1) 
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adj})
            feed_dict.update({self.features: feature})
            feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.eps: eps})
            #feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.basis: basis_matrix})

            feed_dict.update({self.initial_state_c: final_state_c})
            feed_dict.update({self.initial_state_h: final_state_h})

            prob, ll, final_state_c, final_state_h, loss = self.sess.run([self.prob, self.ll, self.final_state_c, self.final_state_h, self.cost],feed_dict=feed_dict )
        
            prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
            prob = np.divide(prob, np.sum(prob))
            k += edges_seq
            
            #probtotal = 0.0
            
            for (u,v) in candidate_edges:
                #print("Debug prob", prob[u][v], u, v)
                probtotal += log(prob[u][v])
        probtotal_end = 0.0
        for (u,v) in edges:
                #print("Debug prob", prob[u][v], u, v)
                probtotal_end += log(prob[u][v])

        with open(hparams.sample_file+ "/reconstruntion_loss.txt", "a") as fw:
            fw.write(str(-np.mean(ll))+"\n")
        with open(hparams.sample_file+ "/prob_derived_end.txt", "a") as fw:
            fw.write(str(probtotal_end)+"\n")
        with open(hparams.sample_file+ "/prob_derived_inter.txt", "a") as fw:
            fw.write(str(probtotal)+"\n")
        with open(hparams.sample_file+ "/elbo.txt", "a") as fw:
            fw.write(str(-np.mean(loss))+"\n")


    def sample_graph(self, hparams, placeholders,s_num,  outdir=None):
        '''
        Args :
            num - int
                10
                number of edges to be sampled
            outdir - string
                output dir
            

        '''
        num_edges = hparams.sample_edge
        edges_seq = int(ceil(num_edges // self.n_seq))
        list_edges = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                    list_edges.append((i, j, 1))
                    list_edges.append((i, j, 2))
                    list_edges.append((i, j, 3))

        #prev_state = sess.run(self.cell.zero_state(1, tf.float32))
        eps = np.random.randn(self.n, self.z_dim, 1) 
        hparams.sample = True
        seen_list = []
        candidate_edges =[ list_edges[i] for i in random.sample(range(len(list_edges)), 1)]
        seen_list.extend(candidate_edges)
        
        adj = np.zeros([self.n, self.n])
        weight_mat = np.zeros([self.n, self.n])
        weight_bin = np.zeros([self.n, self.n, self.bin_dim])

        for (u,v) in candidate_edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
            weight_mat[u][v]+=1
            weight_mat[v][u]+=1

            weight_bin[u][v][weight_mat[u][v]] = 1
            weight_bin[v][u][weight_mat[v][u]] = 1

        final_state_c, final_state_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))
        #self.cell.zero_state(batch_size=1, dtype=tf.float32)
        
        feature = np.zeros([self.n, 1], dtype=np.float)
        for i in range(self.n):
            feature[i][0] =  np.sum(adj[i])//(self.n - 1)

        basis_matrix = basis(adj)

        eps = np.random.randn(self.n, self.z_dim, 1) 
        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj})
        feed_dict.update({self.features: feature})
        feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
        feed_dict.update({self.eps: eps})
        feed_dict.update({self.weight:weight_mat })
        feed_dict.update({self.weight_bin:weight_bin })
        feed_dict.update({self.basis: basis_matrix })
        feed_dict.update({self.initial_state_c: final_state_c})
        feed_dict.update({self.initial_state_h: final_state_h})

        prob, weight, ll, final_state_c, final_state_h = self.sess.run([self.prob, self.w_edge, self.ll, self.final_state_c, self.final_state_h],feed_dict=feed_dict )
        
        problist  = normalise_new(prob, weight)
        
        candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[edges_seq - 1], p=p, replace=False)]
        seen_list.extend(candidate_edges)
        
        while k < num_edges:
            for (u,v) in candidate_edges:
                adj[u][v] = 1
                adj[v][u] = 1
                weight_mat[u][v] += 1
                weight_mat[v][u] += 1

            for i in range(self.n):
                for j in range(self.n):
                    weight_bin[u][v][weight_mat[i][j]] = 1

            for i in range(self.n):
                feature[i][0] =  np.sum(adj[i]) * 1.0 / (self.n - 1)

            basis_matrix = basis(adj)

            eps = np.random.randn(self.n, self.z_dim, 1) 
            
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adj})
            feed_dict.update({self.features: feature})
            feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.eps: eps})
            feed_dict.update({self.basis: basis_matrix})
            feed_dict.update({self.initial_state_c: final_state_c})
            feed_dict.update({self.initial_state_h: final_state_h})

            #prob, ll, final_state_c, final_state_h = self.sess.run([self.prob, self.ll, self.final_state_c, self.final_state_h],feed_dict=feed_dict )
            prob, weight, ll, final_state_c, final_state_h = self.sess.run([self.prob, self.w_edge, self.ll, self.final_state_c, self.final_state_h],feed_dict=feed_dict )
        
            prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
            prob = np.divide(prob, np.sum(prob))

            problist = normalise_new(prob, weight)
            
            candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[min(edges_seq, num_edges - k)], p=p, replace=False)]
            seen_list.extend(candidate_edges) 
            k += edges_seq
        
        for i in range(self.n):
           for j in range(i+1, self.n):
               if adj[i][j] == 1:
                   with open(hparams.sample_file+"approach_2"+str(s_num)+".txt", "a") as fw:
                        fw.write(str(i)+" " + str(j)+ " {}\n")
