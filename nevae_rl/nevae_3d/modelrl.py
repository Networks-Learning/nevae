import tensorflow as tf
from utils import *
from config import SAVE_DIR, VAEGConfig
from cell_3d import VAEGCell
from rlcell_3d import VAEGRLCell
#import tensorflow as tf
import numpy as np
import logging
import copy
import os
import time
import networkx as nx
from collections import defaultdict
from operator import itemgetter
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VAEGRL(VAEGConfig):
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
        self.temperature = hparams.temperature
        self.E = hparams.E
        self.no_traj = hparams.no_traj

        self.adj = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='adj')
        self.features = tf.placeholder(dtype=tf.float32, shape=[self.n, self.d], name='features')
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.k, self.n, self.d], name='input')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[self.n, self.z_dim, 1], name='eps')
        self.coord = tf.placeholder(dtype=tf.float32, shape=[self.n, 3], name='coordinates')
        self.z_coord = tf.placeholder(dtype=tf.float32, shape=[self.n, 2 * self.z_dim], name='z_coord')
        self.features1 = tf.placeholder(dtype=tf.int32, shape=[self.n], name='features1')
        self.weight = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name="weight")
        self.weight_bin = tf.placeholder(dtype=tf.float32, shape=[self.combination, None, hparams.bin_dim], name="weight_bin")


        #For every trajectory
        #self.edges = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='edges') 
        #self.neg_edges = tf.placeholder(dtype=tf.int32, shape=[self.no_traj, None, 2], name='neg_edges') 
        self.all_edges = tf.placeholder(dtype=tf.int32, shape=[self.combination, None, 2], name='all_edges')
        
	self.coord_samples = tf.placeholder(dtype=tf.float32, shape=[self.no_traj, self.n, 3], name='coord_samples')
        self.properties = tf.placeholder(dtype=tf.float32, shape=[self.no_traj], name="properties")
        node_count_tf = tf.fill([1, self.input_size],tf.cast(self.n, tf.float32))
        
        self.cell = VAEGCell(self.adj, self.weight, self.features, self.coord, self.z_dim, self.bin_dim, node_count_tf, self.all_edges)
       
        self.enc_mu, self.enc_sigma, self.z_encoded, coord_mu, coord_sigma = self.cell.call(self.input_data, self.n, self.d, self.k, self.combination, self.z_coord, self.eps, hparams.sample)
	
        
        self.rlcell = VAEGRLCell(self.adj, self.weight, self.features, self.coord, self.z_dim, self.bin_dim, node_count_tf, self.all_edges)
        self.coord_mu, self.coord_sigma = self.rlcell.call(self.input_data, self.n, self.d, self.k, self.combination, self.z_coord, self.z_encoded, hparams.sample)
        print "Debug coord_mu, coord_sigma", self.coord_mu, self.coord_sigma
        print_vars("trainable_variables")
        total_cost = 0.0
        #self.lr = tf.Print(self.lr, [self.lr], message="my lr-values:")
        #self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-06)
        ll = []
        #self.grad = []
        self.grad_placeholder = []
        ll_rl = []
        self.apply_transform_op = []
        tvars = tf.trainable_variables()
        g_vars = [var for var in tvars if 'RL' in var.name]
        print("Debug gvars", g_vars)
        V = []
        ll = []
        ll_rl = []
        w_list = []
        loss = 0.0
        ll_loss = 0.0
        
        for i in range(self.no_traj):
	    ll_temp = self.ll_multivariate_normal(coord_mu, coord_sigma, self.coord_samples[i])
	    ll.append(ll_temp)
            
	    ll_rl_temp = self.ll_multivariate_normal(self.coord_mu, self.coord_sigma, self.coord_samples[i])
            ll_rl.append(ll_rl_temp)

            w_list.append(tf.subtract(ll_rl[i], ll[i]) + self.temperature * self.properties[i] + 1.0)
            ll_loss += (ll_rl[i] - ll[i])
            loss += (ll_rl[i] - ll[i]) + self.temperature * self.properties[i]
            w_total = tf.add_n(w_list)
            w_total = tf.Print(w_total, [w_total], message="my wtotal-values:")
        
        self.ll_loss = ll_loss/ self.no_traj
        self.loss = loss/ self.no_traj
        temp_grad = []
        temp_c_grad = []
        grad_val = []
        grad_c_val =[]
        grad_shape = []
        grad_c_shape = []
        grad_comparison = self.train_op.compute_gradients(self.loss)
        
	for x in range(len(g_vars)):
                        if grad_comparison[x][0] is not None:
                                g = grad_comparison[x]
                        else:
                                g = (tf.fill(tf.shape(g_vars[x]), tf.cast(0.0, tf.float32)), grad_comparison[x][1])
                        #if i == 0:
                        grad_c_val.append(g[0])
                        grad_c_shape.append(g[0].get_shape().as_list())
                        
        for i in range(self.no_traj):
            	grad = self.train_op.compute_gradients(ll_rl[i], var_list=g_vars)
            	w = w_list[i]
		#w = tf.divide(w_list[i], w_total) 
        	w = tf.Print(w, [w], message="my Imp weight-values:")
	 	print "Debug shape w", w.get_shape()	
                for x in range(len(g_vars)):
	    	    if grad[x][0] is not None:
                    		g = grad[x]
                    else:
                    		g = (tf.fill(tf.shape(g_vars[x]), tf.cast(0.0, tf.float32)), grad[x][1])
		    if i == 0:
			    print "Grad shape", g_vars[x], g[0].get_shape()
			    temp_grad.append((w * g[0] / (self.no_traj * 50), g[1]))
			    grad_val.append(w * g[0])
			    grad_shape.append(g[0].get_shape().as_list())
		    else:
			    temp_grad[x] = (tf.add(temp_grad[x][0], w * g[0])/(self.no_traj * 50), g[1])
			    grad_val[x] = tf.add(grad_val[x], w * g[0])
                            #grad_shape.append(g[0].get_shape().as_list())
        print("Debug Grad length", temp_grad, len(temp_grad), len(g_vars))
        self.grad = temp_grad
	self.apply_transform_op = self.train_op.apply_gradients(temp_grad)
        #self.grad = temp_grad
        self.sess = tf.Session()
        #self.error = error
        # We are considering 10 trajectories only


    def ll_multivariate_normal(self, mu, sigma, x_list):
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

		print "Debug shape of all", temp1.get_shape(), temp2.get_shape(), temp3.get_shape()
                ll += -0.5 * temp2 - 0.5 * temp3 - 1.5 * tf.log(2 * 3.14)
            print "Debug LL shpae", tf.reduce_sum(ll).get_shape()
	    return -tf.reduce_sum(ll)

    def label_loss_predict(self, label, predicted_labels):
                loss = 0.0
                #for i in range(self.combination):
                predicted_label = predicted_labels
                predicted_label_resized = tf.reshape(predicted_label, [self.n, self.d])
                predicted_label_exp = tf.exp(tf.minimum(predicted_label_resized, tf.fill([self.n, self.d], 10.0)))
                predicted_label_pos = tf.reduce_sum(tf.multiply(label, predicted_label_exp), axis=1)
                predicted_label_total = tf.reduce_sum(predicted_label_exp, axis=1)
                predicted_label_prob = tf.divide(predicted_label_pos, predicted_label_total)
                ll = tf.reduce_sum(tf.log(tf.add( predicted_label_prob, tf.fill([self.n, ], 1e-9))))
                return ll

    def likelihood_poisson(self, lambda_, x):
            #x_convert = tf.cast(tf.convert_to_tensor([x]), tf.float32)
            x = tf.Print(x, [x], message="My debug_x_tf")
            log_fact_tf = tf.convert_to_tensor([self.log_fact_k[x-1]], dtype=tf.float32)
            return tf.subtract(tf.subtract(tf.multiply(x, tf.log(lambda_ + 1e-09)), lambda_), log_fact_tf)

    def likelihood(self, adj, edges, neg_edges, weight_bin, prob_dict, w_edge, penalty):
            '''
            negative loglikelihood of the edges
            '''
            ll = 0
            k = 0
            with tf.variable_scope('NLL'):
                    dec_mat_temp = tf.reshape(prob_dict, [self.n, self.n])                
                    dec_mat = tf.exp(tf.minimum(dec_mat_temp, tf.fill([self.n, self.n], tf.cast(10.0, dtype=tf.float32))))
                    dec_mat = tf.Print(dec_mat, [dec_mat], message="my decscore values:")
                    min_val = tf.reduce_mean(dec_mat)
                    penalty = tf.exp(penalty)
                    w_edge_resized = tf.reshape(w_edge, [self.n, self.n, self.bin_dim])
                    w_edge_exp = tf.exp(tf.minimum(w_edge_resized, tf.fill([self.n, self.n, self.bin_dim], 10.0)))
                    w_edge_pos = tf.reduce_sum(tf.multiply(weight_bin, w_edge_exp), axis=2)
                    #print "Debug w_edge posscore", w_edge_pos.shape, dec_mat.shape
                    w_edge_total = tf.reduce_sum(w_edge_exp, axis=2)
                    w_edge_score = tf.gather_nd(tf.divide(w_edge_pos, w_edge_total), edges)
                    w_edge_score = tf.Print(w_edge_score, [w_edge_score], message="my w_edge_score values:")
                    #print "Debug w_edge_score", w_edge_score.shape
                    
                    comp = tf.subtract(tf.ones([self.n, self.n], tf.float32), adj)
                    comp = tf.Print(comp, [comp], message="my comp values:")
                    
                    negscore = tf.multiply(comp, dec_mat)
                    negscore = tf.Print(negscore, [negscore], message="my negscore values:")
                    negscore = tf.gather_nd(negscore, neg_edges)
                    negscore_sum = tf.reduce_sum(negscore)
                    
                    posscore = tf.gather_nd(dec_mat, edges)
                    #print "Debug posscore", posscore.shape
                    posscore = tf.Print(posscore, [posscore], message="my posscore values:")
                    pos_weight_score = tf.multiply(posscore, w_edge_score)
                    st = tf.stack([tf.shape(pos_weight_score)[0]])[0]

                    softmax_out = tf.divide(pos_weight_score, negscore_sum)
                    penalty = tf.log(tf.divide(penalty, negscore_sum))
                    comp = tf.Print(comp, [comp], message="my comp values:")

                    ll += tf.reduce_sum(tf.log(tf.add(softmax_out, tf.fill([1, st], 1e-9)))) + penalty
                    ll = tf.Print(ll, [ll], message="My loss")
            
            return (ll)

    def get_trajectories_nevae(self, p_theta, w_theta, edges, weight, n_fill_edges, atom_list):
            indicator = np.ones([self.n, self.bin_dim])
            list_edges = []
            degree = np.zeros(self.n)
            for i in range(self.n):
                for j in range(i+1, self.n):
                    # removing the possibility of hydrogen hydrogen bond and oxigen bond
                    if (atom_list[i] > 1 or atom_list[j] > 1) and (atom_list[i]!=2 or atom_list[j]!=2):
                        list_edges.append((i,j,1))
                        list_edges.append((i,j,2))
                        list_edges.append((i,j,3))

            known_edges = []
            for i in range(self.n):
                # the atom is hydrogen
                if atom_list[i] <= 1:
                    indicator[i][1] = 0
                if atom_list[i] <= 2:
                    indicator[i][2] = 0

            for k in range(self.E):
                (u, v) = edges[k]
                w = weight[u][v]
                degree[u] += w
                degree[v] += w
                if (atom_list[u] - degree[u]) == 0:
                    indicator[u][0] = 0
                if (atom_list[u] - degree[u]) <= 1:
                    indicator[u][1] = 0
                if (atom_list[u] - degree[u]) <= 2:
                    indicator[u][2] = 0

                if (atom_list[v] - degree[v]) == 0:
                    indicator[v][0] = 0
                if (atom_list[v] - degree[v]) <= 1:
                    indicator[v][1] = 0
                if (atom_list[v] - degree[v]) <= 2:
                    indicator[v][2] = 0

                if u < v:
                    list_edges.remove((u, v, 1))
                    list_edges.remove((u, v, 2))
                    list_edges.remove((u, v, 3))
                    known_edges.append((u, v, w))
                else:
                    list_edges.remove((v, u, 1))
                    list_edges.remove((v, u, 2))
                    list_edges.remove((v, u, 3))
                    known_edges.append((v, u, w))

            trial = 0
            adj = np.zeros((self.n, self.n))
            print  "Debug len list edges prev", len(list_edges) 
            G_list = []
            adj_list = []
            G_best = ''
            candidate_edges_list = get_masked_candidate(self.n, list_edges, known_edges, p_theta, w_theta, n_fill_edges, indicator, degree, atom_list)
            
            for candidate_edges in candidate_edges_list:
                adj = np.zeros((self.n, self.n))
                if len(candidate_edges) > 0:
                    candidate_edges_weighted = []
                    for (u, v, w) in candidate_edges:

                        if int(u) < int(v):
                            candidate_edges_weighted.append(str(u) + ' ' + str(v) + ' ' + "{'weight':"+str(w)+"}")
                        else:
                            candidate_edges_weighted.append(str(v) + ' ' + str(u) + ' ' + "{'weight':"+str(w)+"}")
                    
                    G = nx.parse_edgelist(candidate_edges_weighted, nodetype=int)
                    for i in range(self.n):
                        if i not in G.nodes():
                            G.add_node(i)
                    #print "Debug candidate_edges", candidate_edges
                    #if nx.is_connected(G):
                    #    print "Debug graph G is connected"#, candidate_edges
                    #    #candidate_best = candidate_edges
                    #    #G_best = G
	    	    #	#break
            
                    for (u, v, w) in candidate_edges:
                            adj[int(u)][int(v)] = int(w)
                            adj[int(v)][int(u)] = int(w)
                    adj_list.append(adj)
                    G_list.append(G)

            return adj_list, G_list

    '''
    def get_trajectories_nevae(self, p_theta, w_theta, edges, weight, n_fill_edges, atom_list):
            indicator = np.ones([self.n, self.bin_dim])
            list_edges = []
            degree = np.zeros(self.n)
            for i in range(self.n):
                for j in range(i+1, self.n):
                    # removing the possibility of hydrogen hydrogen bond and oxigen bond
                    if (atom_list[i] > 1 or atom_list[j] > 1) and (atom_list[i]!=2 or atom_list[j]!=2):
                        list_edges.append((i,j,1))
                        list_edges.append((i,j,2))
                        list_edges.append((i,j,3))

            known_edges = []
            for i in range(self.n):
                # the atom is hydrogen
                if atom_list[i] <= 1:
                    indicator[i][1] = 0
                if atom_list[i] <= 2:
                    indicator[i][2] = 0

            for k in range(self.E):
                (u, v) = edges[k]
                w = weight[u][v]
                degree[u] += w
                degree[v] += w
                if (atom_list[u] - degree[u]) == 0:
                    indicator[u][0] = 0
                if (atom_list[u] - degree[u]) <= 1:
                    indicator[u][1] = 0
                if (atom_list[u] - degree[u]) <= 2:
                    indicator[u][2] = 0

                if (atom_list[v] - degree[v]) == 0:
                    indicator[v][0] = 0
                if (atom_list[v] - degree[v]) <= 1:
                    indicator[v][1] = 0
                if (atom_list[v] - degree[v]) <= 2:
                    indicator[v][2] = 0

                if u < v:
                    list_edges.remove((u, v, 1))
                    list_edges.remove((u, v, 2))
                    list_edges.remove((u, v, 3))
                    known_edges.append((u, v, w))
                else:
                    list_edges.remove((v, u, 1))
                    list_edges.remove((v, u, 2))
                    list_edges.remove((v, u, 3))
                    known_edges.append((v, u, w))

            trial = 0
            adj = np.zeros((self.n, self.n))
            print  "Debug len list edges prev", len(list_edges) 
            G_list = []
            adj_list = []
            G_best = ''
            candidate_best = []
            while trial < 1:
                #print "Debug degree before calling", degree
                candidate_edges = get_masked_candidate(self.n, list_edges, known_edges, p_theta, w_theta, n_fill_edges, indicator, degree, atom_list)
                #print "Debug candidate edges", len(candidate_edges), candidate_edges
                #print "Debug indicator after calling", indicator
                if len(candidate_edges) > 0:
                    candidate_edges_weighted = []
                    for (u, v, w) in candidate_edges:

                        if int(u) < int(v):
                            candidate_edges_weighted.append(str(u) + ' ' + str(v) + ' ' + "{'weight':"+str(w)+"}")
                        else:
                            candidate_edges_weighted.append(str(v) + ' ' + str(u) + ' ' + "{'weight':"+str(w)+"}")
                    
                    G = nx.parse_edgelist(candidate_edges_weighted, nodetype=int)
                    for i in range(self.n):
                        if i not in G.nodes():
                            G.add_node(i)
                    
                    if nx.is_connected(G):
                        print "Debug graph G is connected", candidate_edges
                        candidate_best = candidate_edges
                        G_best = G
			break
                    else:
                        if len(candidate_best) < len(candidate_edges):
                            candidate_best = candidate_edges
                            G_best = G
                trial += 1
                print("Trial", trial)
            
            for (u, v, w) in candidate_best:
                            adj[int(u)][int(v)] = int(w)
                            adj[int(v)][int(u)] = int(w)

            return adj, G_best
    '''

    def get_trajectories_synthetic(self, p_theta, w_theta, edges, n_fill_edges):

            list_edges = []
            prob = np.reshape(p_theta,(self.n, self.n))
            temp = np.ones([self.n, self.n])
            p_rs = np.exp(np.minimum(prob, 10 * temp))
            denom = np.sum(p_rs)
            p = np.divide(p_rs, denom)
            p_new = []
        
            for i in range(self.n):
                for j in range(i+1, self.n):
                    list_edges.append((i,j,1))
                    p_new.append(p[i][j])

            print("Debug list edges", len(list_edges))
            known_edges = []
            for k in range(self.E):
                (u,v) = edges[k]
                if u < v:
                    list_edges.remove((u, v, 1))
                    p_new.remove(p[u][v])
                    known_edges.append((u, v, 1))
                else:
                    list_edges.remove((v, u, 1))
                    p_new.remove(p[v][u])
                    known_edges.append((v, u, 1))

            p = p_new / np.array(p_new).sum()
            
            print("Debug list_edges", len(list_edges), n_fill_edges) 
            trial = 0
            candidate_edges = []
            G = nx.Graph()
            #for j in range(10):
            adj = np.zeros((self.n, self.n))
            
            while trial < 500:
                    candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[n_fill_edges], p=p, replace=False)]
                    candidate_edges.extend(known_edges)
                    
                    candidate_edges_weighted = []
                    for (u, v, w) in candidate_edges:

                    	if u < v:
                        	candidate_edges_weighted.append(str(u) + ' ' + str(v) + ' ' + "{'weight':"+str(w)+"}")
                    	else:
                        	candidate_edges_weighted.append(str(v) + ' ' + str(u) + ' ' + "{'weight':"+str(w)+"}")

                    G = nx.parse_edgelist(candidate_edges_weighted, nodetype=int)
                    if nx.is_connected(G):
                        for (u, v, w) in candidate_edges:
                            adj[u][v] = w
                            adj[v][u] = w
                        print("Debug trial", trial, len(G.edges()))
                        break
                    trial += 1
                    print("Trial", trial)
	    return adj, G

    def get_trajectories(self, p_theta, w_theta, edges, weight, n_fill_edges, atom_list):

            indicator = np.ones([self.n, self.bin_dim])
            edge_mask = np.ones([self.n, self.n])
            degree = np.zeros(self.n)
            known_edges = []

            for k in range(self.E):
                (u,v) = edges[k]
                edge_mask[u][v] = 0
                edge_mask[v][u] = 0
                degree[u]+=weight[u][v]
                degree[v]+=weight[v][u]
                known_edges.append((u,v,weight[u][v]))
                if (4 - degree[u]) == 0:
                    indicator[u][0] = 0
                if (4 - degree[u]) <= 1:
                    indicator[u][1] = 0
                if (4 - degree[u]) <= 2:
                    indicator[u][2] = 0

                if (4 - degree[v]) == 0:
                    indicator[v][0] = 0
                if (4 - degree[v]) <= 1:
                    indicator[v][1] = 0
                if (4 - degree[v]) <= 2:
                    indicator[v][2] = 0
            #'''
            trial = 0
            candidate_edges = []
            G = nx.Graph()

            while trial < 50:
                candidate_edges = get_masked_candidate_new(p_theta, w_theta, n_fill_edges, atom_list, indicator, edge_mask, degree)
                candidate_edges.extend(known_edges)
                G = nx.Graph()
                G.add_nodes_from(range(self.n))
                G.add_weighted_edges_from(candidate_edges)
                if nx.is_connected(G):
                    print("Debug trial", trial)
                    break
                trial += 1
                print("Trial", trial)
            return candidate_edges, G


    def initialize(self):
        logger.info("Initialization of parameters")
        # self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

    
    def restore(self, savedir):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 20)
        ckpt = tf.train.get_checkpoint_state(savedir)
        if ckpt == None or ckpt.model_checkpoint_path == None:
            self.initialize()
        else:
            print("Load the model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def partial_restore(self, copydir):
	saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 20 )
        self.initialize()
        print("Debug all", tf.global_variables())
        var_old = [v for v in tf.global_variables() if "RL" not in v.name]
        print("Debug var_old", var_old)
        saver = tf.train.Saver(var_old)
        ckpt = tf.train.get_checkpoint_state(copydir)

        #print_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path, tensor_name='', all_tensors='')
        print("Load the model from {}".format(ckpt.model_checkpoint_path))
        #print_tensors_in_checkpoint_file(ckpt, all_tensors=True, tensor_name='')
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        var_new = [v for v in tf.global_variables() if ("RL" in v.name and "Poisson" in v.name) ]
	for v in var_new:
            v_old_temp = [v_old for v_old in tf.global_variables() if v_old.name == v.name.replace("RL", "") ]
            if len(v_old_temp) == 0:
                continue
            v_old = v_old_temp[0]
            print("v_old", v_old.value(), v_old.name)
            #if v_old  in var_old
            assign = tf.assign(v, v_old)
            self.sess.run(assign)
            #v = tf.Variable(v.name.replace("RL", ""))
            print("v_new", v, v.name)
 

    def copy_weight(self, copydir):
        self.initialize()
        print("Debug all", tf.global_variables())
        var_old = [v for v in tf.global_variables() if "RL" not in v.name]
        print("Debug var_old", var_old)
        saver_old = tf.train.Saver(var_old, max_to_keep=20)
        ckpt = tf.train.get_checkpoint_state(copydir)
        
        #print_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path, tensor_name='', all_tensors='')
        print("Load the model from {}".format(ckpt.model_checkpoint_path))
        #print_tensors_in_checkpoint_file(ckpt, all_tensors=True, tensor_name='')
        saver_old.restore(self.sess, ckpt.model_checkpoint_path)
        var_new = [v for v in tf.global_variables() if "RL" in v.name]
        print("Debug var_new", var_new)
        for v in var_new:
            v_old_temp = [v_old for v_old in tf.global_variables() if v_old.name == v.name.replace("RL", "") ]
            if len(v_old_temp) == 0:
		continue
	    v_old = v_old_temp[0]
	    print("v_old", v_old.value(), v_old.name) 
            #if v_old  in var_old
            assign = tf.assign(v, v_old)
	    self.sess.run(assign)
	    #v = tf.Variable(v.name.replace("RL", ""))
	    print("v_new", v, v.name)
    	saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)


    def train(self, placeholders, hparams, adj, weight, weight_bin, features, edges, all_edges, features1, coord):
        savedir = hparams.out_dir
        lr = hparams.learning_rate
        dr = hparams.dropout_rate
        decay = hparams.decay_rate

        f1 = open(hparams.out_dir + '/iteration.txt', 'r')
        iter1 = int(f1.read().strip())
        iteration = iter1
        
        # training
        num_epochs = hparams.num_epochs
        create_dir(savedir)
        ckpt = tf.train.get_checkpoint_state(savedir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        
        start_before_epoch = time.time()
        importance_weight = 0.0
        tvars = tf.trainable_variables()
        g_vars = [var for var in tvars if 'RL' in var.name]
        print("Debug g_vars", g_vars)
        
        grad_local = []
        for x in range(len(g_vars)):
            a = np.zeros(shape=g_vars[x].get_shape().as_list(), dtype=float)
            #a.fill(0.0)
            print("Debug a", a, a.shape)
            grad_local.append(a)
        print("Debug gradlocal", grad_local, g_vars[0].get_shape().as_list())
        all_edges_local = []
        for i in range(self.n):
            for j in range(self.n):
                all_edges_local.append((i,j))
        prev_loss = 10000
	epoch = 0
        
        
        #print "Debug props logp", mean_logp, std_logp, "SAS: ", mean_sas, std_sas, "Cycle :", mean_cycle, std_cycle
       
        while (epoch < num_epochs):
        #for epoch in range(num_epochs):
            for i in range(len(adj)):
            	print "Debug inside loop", epoch
                start = time.time()
                start1 = time.time()
                feed_dict = construct_feed_dict(lr, dr, self.k, self.n, self.d, decay, placeholders)
                # we will sample 50 z values here
                count = 0
                total_train_loss = 0.0
                total_ll_loss = 0.0
		
                #while count < 50:
                eps = np.random.randn(self.n, self.z_dim, 1)  
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.eps: eps})
                feed_dict.update({self.all_edges: [all_edges_local]})
		feed_dict.update({self.adj:adj[i]})
		feed_dict.update({self.weight:weight[i]})
		feed_dict.update({self.features: features[i]})
                feed_dict.update({self.features1: features1[i]})
                feed_dict.update({self.weight_bin: weight_bin[i]})
		feed_dict.update({self.coord:coord[i]})

                    
		enc_mu, enc_sigma, _ = self.sess.run([self.enc_mu, self.enc_sigma, self.z_encoded], feed_dict=feed_dict)
              
                while count < 10:
			list_adj = []
                	list_prop = []
                	list_edge = []
                	list_neg_edge = []
			eps = np.random.randn(self.n, self.z_dim, 1)
                    	#z_encoded = sample_gaussian(eps, enc_mu, enc_sigma)
		    	temp_stack = []
                        for j in range(self.n):
                	    temp_stack.append(np.matmul(enc_sigma[j], eps[j]))
            	    	z_encoded = np.add(enc_mu, np.array(temp_stack))

                        z_coord_modified = neighbor(z_encoded, weight[i], self.z_dim)
		    	feed_dict.update({self.eps:z_encoded})
			feed_dict.update({self.z_coord:z_coord_modified})
                        #print "Debug data z_cocord", z_coord_modified.shape, z_encoded.shape
			#print "Debug feed_dict", feed_dict
                        coord_mu, coord_sigma = self.sess.run([self.coord_mu, self.coord_sigma], feed_dict=feed_dict)
		        print "Debug coord_mu", coord_mu, coord_sigma	
			#hparams.sample = True                    
			coord_samples = []
		    	
			#coord_samples = np.random.multivariate_normal()
			for x in range(self.no_traj):
				list_adj.append(adj[i])
				temp_stack = []
                                coords = get_coordinates(coord_mu, coord_sigma, adj[i])
                                coord_samples.append(coords)

                        properties = []	
		    	for j in range(hparams.no_traj):
                        	energy = compute_cost_energy(weight[i], coord_samples[j], features1[i])
                        	#energy_list.append(energy)
                                #print "energy", energy
               			properties.append(energy) 
                    
		    	index_list = np.argsort(properties)[:hparams.no_traj]
                    
		    
		    	properties_new = [properties[x] for x in index_list]
		    	coord_samples_new = [coord_samples[x] for x in index_list]
 
                    	feed_dict.update({self.properties:properties_new})
                    	feed_dict.update({self.coord_samples: coord_samples_new})
                    	_, grad,  train_loss, ll_loss = self.sess.run([self.apply_transform_op,  self.grad, self.loss, self.ll_loss], feed_dict=feed_dict)
                    
                    	print("Time size of graph", len(tf.get_default_graph().get_operations()))
                    	properties_original = [-(1000.0 - x) for x in properties_new]
                    	total_train_loss += train_loss 
                   	total_ll_loss += ll_loss
                        
                        for x in range(len(coord_samples_new)): 
                    	    	print("LOSS ", properties_original[x])
				#print("LOSS ",count, train_loss, ll_loss,  properties_original, coord_samples_new[x])
                            	for j in range(self.n):
                                	print features1[i][j], coord_samples_new[x][j]

				
                    	#print("Grad", grad)
                    	end2 = time.time()
                    	count += 1
            iteration += 1
            prev_loss = train_loss
            epoch += 1
            if iteration % hparams.log_every == 0 and iteration > 0:
                #print(train_loss)
                print("{}/{}(epoch {}), train_loss = {}, ll_loss={}".format(iteration, num_epochs, epoch + 1, total_train_loss, total_ll_loss))
                checkpoint_path = os.path.join(savedir, 'model.ckpt')
                saver.save(self.sess, checkpoint_path, global_step=iteration)
                logger.info("model saved to {}".format(checkpoint_path))
                end = time.time()
                print("Time taken for a batch: ",end - start, end2 - start1)
        end_after_epoch = time.time()
        print("Time taken to completed all epochs", -start_before_epoch + end_after_epoch)
        f1 = open(hparams.out_dir+'/iteration.txt','w')
        f1.write(str(iteration))
        

