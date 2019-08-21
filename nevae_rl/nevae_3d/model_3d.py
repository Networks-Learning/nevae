from utils import *
from config import SAVE_DIR, VAEGConfig
from cell_3d import VAEGCell
from math import log
import tensorflow as tf
import numpy as np
import logging
import copy
import os
import time
import networkx as nx
from collections import defaultdict
from operator import itemgetter
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VAEG(VAEGConfig):
    def __init__(self, hparams, placeholders, num_nodes, num_features, log_fact_k, input_size, istest=False):
        self.features_dim = num_features
        self.input_dim = num_nodes
        self.dropout = placeholders['dropout']
        self.k = hparams.random_walk
        self.lr = placeholders['lr']
        self.decay = placeholders['decay']
        self.n = num_nodes
        self.d = num_features
        self.z_dim = hparams.z_dim
        self.bin_dim = hparams.bin_dim
        self.mask_weight = hparams.mask_weight
        self.log_fact_k = log_fact_k
        self.neg_sample_size = hparams.neg_sample_size
        self.input_size = input_size
        self.combination = hparams.node_sample * hparams.bfs_sample

        def neg_loglikelihood(prob_dicts, w_edges):
            '''
            negative loglikelihood of the edges
            '''
            ll = 0
            k = 0
            with tf.variable_scope('NLL'):
                for i in range(self.combination):
                    prob_dict = prob_dicts[i]
                    w_edge = w_edges[i]
                    prob_dict = tf.Print(prob_dict, [prob_dict], message="my prob dict values:")
                    print("Debug prob dict shape", tf.shape(prob_dict))
                    prob_dict_resized = tf.reshape(prob_dict, [-1])
                    prob_dict_resized = tf.Print(prob_dict_resized, [prob_dict_resized], message="my prob dict resized values:")
                    w_edge_size = tf.stack([tf.shape(w_edge)[0]])[0]
                    w_edge_size = tf.Print(w_edge_size, [w_edge_size], message="my size values:")
                    print("Debug w_edge_shape", tf.shape(w_edge), w_edge.get_shape(), tf.stack([tf.shape(w_edge)[0]])[0])
                    w_edge_resized = tf.reshape(w_edge, [-1, self.bin_dim])
                    if self.neg_sample_size > 0:
                        w_edge_resized = tf.reshape(w_edge[:-self.bin_dim * self.neg_sample_size], [-1, self.bin_dim])
                    
                    w_edge_size_r = tf.stack([tf.shape(w_edge_resized)[0]])[0]
                    w_edge_size_r = tf.Print(w_edge_size_r, [w_edge_size_r], message="my size values r:")
                    w_edge_exp = tf.exp(tf.minimum(w_edge_resized, tf.fill([w_edge_size_r, self.bin_dim], 10.0)))
                    w_edge_pos = tf.reduce_sum(tf.multiply(self.weight_bin[i], w_edge_exp), axis=1)
                    w_edge_total = tf.reduce_sum(w_edge_exp, axis=1)
                    w_edge_score = tf.divide(w_edge_pos, w_edge_total)
                    w_edge_score = tf.Print(w_edge_score, [w_edge_score], message="my w_edge_score values:")
                    
                    prob_dict_resized_shape = tf.stack([tf.shape(prob_dict_resized)[0]])[0]
                    prob_dict_resized_shape = tf.Print(prob_dict_resized_shape, [prob_dict_resized_shape], message="my prob dict size values:")
                    prob_dict_exp = tf.exp(tf.minimum(prob_dict_resized, tf.fill([prob_dict_resized_shape], 10.0)))
                    prob_dict_exp = tf.Print(prob_dict_exp, [prob_dict_exp], message="my decscore values:")
                    pos_score = prob_dict_exp

                    if self.neg_sample_size > 0:
                        pos_score = prob_dict_exp[:-self.neg_sample_size]

                    st = tf.stack([tf.shape(pos_score)[0]])[0]
                    st = tf.Print(st, [st], message="my st values:")
                    pos_score = tf.Print(pos_score, [pos_score], message="my posscore values:")
                    #pos_weight_score = tf.multiply(tf.reshape(pos_score,[st, 1]), w_edge_score)
                    pos_weight_score = tf.multiply(pos_score, tf.reshape(w_edge_score,[1,-1]))
                    neg_score = tf.cumsum(prob_dict_exp , reverse=True)
                    if self.neg_sample_size > 0:
                        neg_score = tf.cumsum(prob_dict_exp[1:] , reverse=True)[:-self.neg_sample_size + 1]
                    softmax_out = tf.divide(pos_weight_score, neg_score)

                    ll += tf.reduce_sum(tf.log(tf.add(softmax_out, tf.fill([1, st], 1e-9))))
                    #ll = tf.reduce_sum(tf.log(tf.add(tf.multiply(self.adj, softmax_out), tf.fill([self.n,self.n], 1e-9))))
                ll = ll / self.combination
                ll = tf.Print(ll, [ll], message="My LL loss")

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
                k = tf.fill([self.n], tf.cast(self.z_dim, tf.float32))
                temp_stack = []
                for i in range(self.n):
                    temp_stack.append(tf.reduce_prod(tf.square(debug_sigma[i])))
                third_term = tf.log(tf.add(tf.stack(temp_stack),tf.fill([self.n],1e-09)))
                return 0.5 * tf.add(tf.subtract(tf.add(first_term ,second_term), k), third_term)
       
        def ll_poisson(lambda_, x):
            #x_convert = tf.cast(tf.convert_to_tensor([x]), tf.float32)
            x = tf.Print(x, [x], message="my debug_x_tf")
            log_fact_tf = tf.convert_to_tensor([self.log_fact_k[x-1]], dtype=tf.float32)
            return -tf.subtract(tf.subtract(tf.multiply(x, tf.log(lambda_ + 1e-09)), lambda_), log_fact_tf)

        def ll_multivariate_normal(mu, sigma, x_list):
            ll = 0.0
            for i in range(self.n):
		x = x_list[i]
                diff = tf.reshape(tf.subtract(x, mu[i]), [3,1])
                sigma_local = sigma[i]
                sigma_local = tf.Print(sigma_local, [sigma_local], message="my debug sigma_local tf")
                sigma_inv = tf.linalg.inv(sigma_local)
                sigma_inv = tf.Print(sigma_inv, [sigma_inv], message="my debug sigma_inv tf")
 
                diff_trans = tf.transpose(diff)
                det = tf.linalg.det(sigma[i])
                det = tf.Print(det, [det], message="my debug det")

                temp1 = tf.matmul(diff_trans, sigma_inv)
                temp1 = tf.Print(temp1, [temp1], message="my debug temp1")

        	temp2 = tf.matmul(temp1, diff)
                temp2 = tf.Print(temp2, [temp2], message="my debug temp2")

                temp3 = tf.log(tf.maximum(det, 0.0) + 1e-09)
                temp3 = tf.Print(temp3, [temp3], message="my debug temp3")

                ll += -0.5 * temp2 - 0.5 * temp3 - 1.5 * tf.log(2 * 3.14)
            return -ll

        
        def label_loss_predict(label, predicted_labels, label1):
                loss = 0.0
                #for i in range(self.combination):
                predicted_label = predicted_labels
                #predicted_label[0] = tf.Print(predicted_label[0], [predicted_label[0]], message="My predicted_value")
                predicted_label_resized = tf.reshape(predicted_label, [self.n, self.d])
                predicted_label_exp = tf.exp(tf.minimum(predicted_label_resized, tf.fill([self.n, self.d], 10.0)))
                predicted_label_pos = tf.reduce_sum(tf.multiply(label, predicted_label_exp), axis=1)
                predicted_label_total = tf.reduce_sum(predicted_label_exp, axis=1)
                predicted_label_prob = tf.divide(predicted_label_pos, predicted_label_total)
                ll = tf.reduce_sum(tf.log(tf.add( predicted_label_prob, tf.fill([self.n, ], 1e-9))))
                return -ll
                

	def get_lossfunc(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma, dec_out, w_edge, label, lambda_n, lambda_e, coor_mu, coor_sigma, coord):
            
            kl_loss = kl_gaussian(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma)  # KL_divergence loss
            likelihood_loss = neg_loglikelihood(dec_out, w_edge)  # Cross entropy loss
            self.ll = likelihood_loss
            self.kl = kl_loss
            
            lambda_e = tf.Print(lambda_e, [lambda_e], message="My edge_lambda")
            lambda_n = tf.Print(lambda_n, [lambda_n], message="My node_lambda")

            #print("Debug self count", self.count, self.edges[self.count])
            edgeprob = ll_poisson(lambda_e, tf.cast(tf.subtract(tf.shape(self.edges[0])[0], self.neg_sample_size), tf.float32))
            nodeprob = ll_poisson(lambda_n, tf.cast(tf.convert_to_tensor([self.n]), tf.float32))
            edgeprob = tf.Print(edgeprob, [edgeprob], message="My edge_prob_loss")
            nodeprob = tf.Print(nodeprob, [nodeprob], message="My node_prob_loss")
            #label = tf.Print(label, [label], message="My label_value")
            label_loss = label_loss_predict(self.features, label, self.features1)
            label_loss = tf.Print(label_loss, [label_loss], message="My label_loss")
            
            x_loss = ll_multivariate_normal(coor_mu, coor_sigma, coord)
	    x_loss = tf.Print(x_loss, [x_loss], message="My x_loss")
	    kl_loss_1 = tf.reduce_mean(kl_loss)
	    kl_loss_1 = tf.Print(kl_loss_1, [kl_loss_1], message="My KL_loss")
            
	    loss_1 = kl_loss_1 + label_loss + x_loss
            loss_1 = tf.Print(loss_1, [loss_1], message="My label_loss KL")
            
	    likelihood_loss = tf.Print(likelihood_loss, [likelihood_loss], message="My likelihood_loss") 
	    total_loss = loss_1 + tf.reduce_mean(edgeprob + nodeprob + likelihood_loss)
            #return tf.reduce_mean(kl_loss) + edgeprob + nodeprob + likelihood_loss
            total_loss = tf.Print(total_loss, [total_loss], message="My total_loss")
	    return total_loss
            

        self.adj = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='adj')
        self.features = tf.placeholder(dtype=tf.float32, shape=[self.n, self.d], name='features')
        self.coord = tf.placeholder(dtype=tf.float32, shape=[self.n, 3], name='coordinates')
        self.z_coord = tf.placeholder(dtype=tf.float32, shape=[self.n, 2 * self.z_dim], name='z_coord')
	self.features1 = tf.placeholder(dtype=tf.int32, shape=[self.n], name='features1')
        self.weight = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name="weight")
        self.weight_bin = tf.placeholder(dtype=tf.float32, shape=[self.combination, None, hparams.bin_dim], name="weight_bin")
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.k, self.n, self.d], name='input')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[self.n, self.z_dim, 1], name='eps')
        #self.neg_index = tf.placeholder(dtype=tf.int32,shape=[None], name='neg_index')
        self.edges = tf.placeholder(dtype=tf.int32, shape=[self.combination, None, 2], name='edges') 
        self.count = tf.placeholder(dtype=tf.int32)

        #node_count = [len(edge_list) for edge_list in self.edges]
        print("Debug Input size", self.input_size)
        node_count_tf = tf.fill([1, self.input_size],tf.cast(self.n, tf.float32))
        node_count_tf = tf.Print(node_count_tf, [node_count_tf], message="My node_count_tf")
        print("Debug size node_count", node_count_tf.get_shape())
        
        #tf.convert_to_tensor(node_count, dtype=tf.int32)
        self.cell = VAEGCell(self.adj, self.weight, self.features, self.coord, self.z_dim, self.bin_dim, node_count_tf, self.edges)
        print("Debug cell initialization")
	self.c_x, enc_mu, enc_sigma, debug_sigma,dec_out, prior_mu, prior_sigma, z_encoded, w_edge, label, coor_mu, coor_sigma, lambda_n, lambda_e = self.cell.call(self.input_data, self.n, self.d, self.k, self.combination, self.z_coord, self.eps, hparams.sample)
        self.prob = dec_out
        self.z_encoded = z_encoded
        self.enc_mu = enc_mu
        self.enc_sigma = enc_sigma
        self.w_edge = w_edge
        self.label = label
        self.lambda_n = lambda_n
        self.lambda_e = lambda_e
        self.cost = get_lossfunc(enc_mu, enc_sigma, debug_sigma, prior_mu, prior_sigma, dec_out, w_edge, label, lambda_n, lambda_e, coor_mu, coor_sigma, self.coord)

        print_vars("trainable_variables")
        # self.lr = tf.Variable(self.lr, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.grad = self.train_op.compute_gradients(self.cost)
        self.grad_placeholder = [(tf.placeholder("float", shape=gr[1].get_shape()), gr[1]) for gr in self.grad]
        self.apply_transform_op = self.train_op.apply_gradients(self.grad)

        #self.lr = tf.Variable(self.lr, trainable=False)
        self.sess = tf.Session()

    def initialize(self):
        logger.info("Initialization of parameters")
        #self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

    def restore(self, savedir):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(savedir)
        if ckpt == None or ckpt.model_checkpoint_path == None:
            self.initialize()
        else:    
            print("Load the model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        #self.sess.close()
	#print("Variables", tf.global_variables())

    def train(self, placeholders, hparams, adj,weight, weight_bin, features, edges, neg_edges, features1, coord):
        savedir = hparams.out_dir
        lr = hparams.learning_rate
        dr = hparams.dropout_rate
        decay = hparams.decay_rate

        f1 = open(hparams.out_dir+'/iteration.txt','r')
        iteration = int(f1.read().strip())
        # training
        num_epochs = hparams.num_epochs
        create_dir(savedir)
        ckpt = tf.train.get_checkpoint_state(savedir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)

        start_before_epoch = time.time()
        for epoch in range(num_epochs):
            start = time.time()
	    train_loss_total = 0.0
	    
            for i in range(len(adj)):
                if len(edges[i]) == 0:
                    continue
                # Learning rate decay
                feed_dict = construct_feed_dict(lr, dr, self.k, self.n, self.d, decay, placeholders)
                feed_dict.update({self.adj: adj[i]})
                eps = np.random.randn(self.n, self.z_dim, 1)  
                
                feed_dict.update({self.features: features[i]})
                feed_dict.update({self.features1: features1[i]})
                feed_dict.update({self.weight_bin: weight_bin[i]})
                feed_dict.update({self.weight: weight[i]})
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.eps: eps})
		feed_dict.update({self.coord: coord[i]})
		# z_coord =                 
		
		neg_indices = np.random.choice(range(len(neg_edges[i])), hparams.neg_sample_size, replace=False)
                combined_edges = []
                neg_edges_to_be_extended = [neg_edges[i][index] for index in neg_indices]
                copy_edge = copy.deepcopy(edges[i])
                
		for j in range(len(edges[i])):
                    copy_edge[j].extend(neg_edges_to_be_extended)
                    
                print("Debug feed edges", i, len(edges[i][0]), len(copy_edge[0]))
                feed_dict.update({self.edges:copy_edge})
		feed_dict.update({self.z_coord:np.zeros((self.n, self.z_dim * 2))})
                z_encoded, cx = self.sess.run([self.z_encoded, self.c_x], feed_dict=feed_dict)
	        z_coord_modified = neighbor(z_encoded, weight[i], self.z_dim)	
		feed_dict.update({self.z_coord:z_coord_modified})
		train_loss, _, probdict, lambda_e, lambda_n, label= self.sess.run([self.cost, self.apply_transform_op, self.prob, self.lambda_e, self.lambda_n, self.label], feed_dict=feed_dict)
                #iteration += 1
		train_loss_total += train_loss
		print( "Lambda_e, lambda_n, loss : ", lambda_e, lambda_n, train_loss )
	    train_loss_total /= len(adj)
	    iteration += 1
	    print("Avg train_loss_total", train_loss_total) 
	    if iteration % hparams.log_every == 0 and iteration > 0:
		#print(train_loss)
                print("{}/{}(epoch {}), train_loss = {}".format(iteration, num_epochs, epoch + 1, train_loss_total))
                checkpoint_path = os.path.join(savedir, 'model.ckpt')
                saver.save(self.sess, checkpoint_path, global_step=iteration)
                logger.info("model saved to {}".format(checkpoint_path))

            end = time.time()
            print("Time taken for a batch: ",end - start )
        end_after_epoch = time.time()
        print("Time taken to completed all epochs", -start_before_epoch + end_after_epoch)
        f1 = open(hparams.out_dir+'/iteration.txt','w')
        f1.write(str(iteration))

    
