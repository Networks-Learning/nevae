import tensorflow as tf
from utils import *
from config import SAVE_DIR, VAEGConfig
from cell import VAEGCell
from rlcell import VAEGRLCell
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
        
        #For every trajectory
        self.edges = tf.placeholder(dtype=tf.int32, shape=[self.no_traj, None, 2], name='edges') 
        self.weight_bin = tf.placeholder(dtype=tf.float32, shape=[self.no_traj, self.n, self.n, hparams.bin_dim], name="weight_bin")
        self.neg_edges = tf.placeholder(dtype=tf.int32, shape=[self.no_traj, None, 2], name='neg_edges') 
        self.all_edges = tf.placeholder(dtype=tf.int32, shape=[self.combination, None, 2], name='all_edges')
        
	# for the time being 5 trajectories are in action
        self.trajectories = tf.placeholder(dtype=tf.float32, shape=[self.no_traj, self.n, self.n], name="trajectories")
        self.properties = tf.placeholder(dtype=tf.float32, shape=[self.no_traj], name="properties")
        self.n_fill_edges = tf.placeholder(dtype=tf.int32)
        self.n_edges = tf.placeholder(dtype=tf.float32)
        self.penalty = tf.placeholder(shape=[self.no_traj],dtype=tf.float32)

        #self.known_edges = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='known_edges') 
        #node_count = [len(edge_list) for edge_list in self.edges]
        #print("Debug Input size", self.input_size)
        
	node_count_tf = tf.fill([1, self.input_size],tf.cast(self.n, tf.float32))
        #node_count_tf = tf.Print(node_count_tf, [node_count_tf], message="My node_count_tf")
        #print("Debug size node_count", node_count_tf.get_shape())
        #tf.convert_to_tensor(node_count, dtype=tf.int32)
        
	self.cell = VAEGCell(self.adj, self.features, self.z_dim, self.bin_dim, node_count_tf, self.all_edges)
        self.c_x, dec_out, z_encoded, w_edge, label, lambda_n, lambda_e = self.cell.call(self.input_data, self.n, self.d, self.k, self.combination, self.eps, hparams.sample)
        self.prob = dec_out
        
	#print('Debug', dec_out.shape)
        self.z_encoded = z_encoded
        #self.enc_mu = enc_mu
        #self.enc_sigma = enc_sigma
	self.w_edge = w_edge
        
	#self.label = label
        #self.lambda_n = lambda_n
        #self.lambda_e = lambda_e
        #adj, weight, features, z_dim, bin_dim, node_count, edges, enc_mu, enc_sigma
        self.rlcell = VAEGRLCell(self.adj, self.features, self.z_dim, self.bin_dim, self.all_edges)
        #self, adj, weight, features, z_dim, bin_dim, enc_mu, enc_sigma, edges, index
        self.rl_dec_out, self.rl_w_edge, self.lambda_e, self.label = self.rlcell.call(self.input_data, self.n, self.d, self.k, self.combination, self.eps, hparams.sample)
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
            #if self.properties[i] == 0:
            #    continue
            ll_temp = self.likelihood(self.trajectories[i], self.edges[i], self.neg_edges[i], self.weight_bin[i], self.prob[0], self.w_edge[0], self.penalty[i])
            #ll_temp = tf.Print(ll_temp, [ll_temp], message="my ll-values:")
            ll_poisson = self.likelihood_poisson(lambda_e, self.n_edges)
            label_pred = self.label_loss_predict(self.features, label)
            #label_pred = tf.Print(label_pred, [label_pred], message="my label-ll-values:")
            ll.append(ll_temp + ll_poisson + label_pred)

            ll_rl_temp = self.likelihood(self.trajectories[i], self.edges[i], self.neg_edges[i], self.weight_bin[i], self.rl_dec_out[0], self.rl_w_edge[0], self.penalty[i])
            #ll_rl_temp = tf.Print(ll_rl_temp,[ll_rl_temp], message="my ll_rl-values:")
            ll_rl_poisson = self.likelihood_poisson(self.lambda_e, self.n_edges)
            label_pred_rl = self.label_loss_predict(self.features, self.label)
            #label_pred_rl = tf.Print(label_pred_rl, [label_pred_rl], message="my label-ll-rl-values:")

            ll_rl.append(ll_rl_temp + ll_rl_poisson + label_pred_rl)

            # w_list.append(self.temperature * tf.subtract(ll_rl[i], ll[i])+self.properties[i])
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
		
                for x in range(len(g_vars)):
	    	    if grad[x][0] is not None:
                    		g = grad[x]
                    else:
                    		g = (tf.fill(tf.shape(g_vars[x]), tf.cast(0.0, tf.float32)), grad[x][1])
		    if i == 0:
			    temp_grad.append((w * g[0] / (self.no_traj * 50), g[1]))
			    grad_val.append(w * g[0])
			    grad_shape.append(g[0].get_shape().as_list())
		    else:
			    temp_grad[x] = (tf.add(temp_grad[x][0], w * g[0])/(self.no_traj * 50), g[1])
			    grad_val[x] = tf.add(grad_val[x], w * g[0])
                            #grad_shape.append(g[0].get_shape().as_list())
        print("Debug Grad length", len(temp_grad), len(g_vars))
        self.grad = temp_grad
	self.apply_transform_op = self.train_op.apply_gradients(temp_grad)
        #self.grad = temp_grad
        self.sess = tf.Session()
        #self.error = error
        # We are considering 10 trajectories only

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
            G_list = []
            adj_list = []
            G_best = ''
            for j in range(1000):
                prob = np.reshape(p_theta, [self.n, self.n])
		w_edge = np.reshape(w_theta, [self.n, self.n, 3])
		edges = self.get_masked_candidate_with_atom_ratio_new(prob, w_edge, atom_count=atom_list, num_edges=n_fill_edges, hde=1)
                G = nx.parse_edgelist(edges, nodetype=int)
                if nx.is_connected(G): 
                        print "Connected"
                        for (u, v) in G.edges():
                            adj[int(u)][int(v)] = 1
			    #int(G[u][v]["weight"])
                            adj[int(v)][int(u)] = 1
			    #int(G[u][v]["weight"])
                        adj_list.append(adj)
                        G_list.append(G)

            #rest = range(self.n)
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

                    if nx.is_connected(G): 
                        for (u, v, w) in candidate_edges:
                            adj[int(u)][int(v)] = int(w)
                            adj[int(v)][int(u)] = int(w)
                        adj_list.append(adj)
                        G_list.append(G)
            return adj_list, G_list


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

    def train(self, placeholders, hparams, adj, weight, weight_bin, weight_bin1, features, edges, all_edges, features1, atom_list):
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

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)
        
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
                i = 0
            	#print "Debug inside loop", epoch
                start = time.time()
                start1 = time.time()
                feed_dict = construct_feed_dict(lr, dr, self.k, self.n, self.d, decay, placeholders)
                # we will sample 50 z values here
                count = 0
                total_train_loss = 0.0
                total_ll_loss = 0.0
                while count < 30:
                    eps = np.random.randn(self.n, self.z_dim, 1)  
                    feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                    feed_dict.update({self.eps: eps})
                    feed_dict.update({self.all_edges: [all_edges_local]})
                    list_adj = []
                    list_prop = []
                    list_edge = []
                    list_neg_edge = []
                    prob, w_edge, rl_prob, rl_w_edge, lambda_e, z_encoded, label = self.sess.run([self.prob, self.w_edge, self.rl_dec_out, self.rl_w_edge, self.lambda_e,  self.z_encoded, self.label], feed_dict=feed_dict)
                    features, atom_list = self.getatoms(self.n, label)
                    if len(atom_list) == 0:
                        print "getatom not satisfied bad Z"
                        end2 = time.time()
                        continue
                    max_edges_possible = int(sum(atom_list)/2)
                    n_edges = max_edges_possible + 1
                
                    while(n_edges > max_edges_possible or n_edges < (self.n - 1) ):
                        n_edges = np.random.poisson(lambda_e) 
                
                    end1 = time.time()
                    weights = []
                    weight_bins = []
                    properties = []
                    pos_edges = []
                    neg_edges = []
                    list_penalty = []
                    qed_list = []
                    t_list, G_list = self.get_trajectories_nevae(rl_prob, rl_w_edge, edges[i][0], weight[i], n_edges - self.E, atom_list)
                    edge_len = []
                    for j in range(len(t_list)):
                        t = t_list[j]
                        G = G_list[j]   
                        qed = compute_cost_qed(G, hparams.out_dir+"temp.txt")
                        qed_list.append(qed)
               		properties.append(qed)
                        edge_len.append(len(G.edges()))
                    index_list = np.argsort(properties)[:hparams.no_traj]
		    if len(index_list) < hparams.no_traj or properties[index_list[0]] == 2.0 :
			continue
		    max_edge = max(edge_len)
		    properties_new = []
                    candidate_edges = []
		    for j in range(hparams.no_traj):
                        index = index_list[j]
                        t = t_list[index]
                        G = G_list[index]
                        rl_prob_reshape = np.reshape(rl_prob, [self.n, self.n])
                        minval = min(rl_prob[0])
                        penalty = 0.0
                        penalty_index = np.unravel_index(np.argmin(rl_prob_reshape, axis=None), rl_prob_reshape.shape)
                    
                        penalty_edges = []
                        if len(G.edges())< max_edge:
                            diff = max_edge - len(G.edges())
                            while diff > 0:
                                penalty += penalty
                                penalty_edges.append(penalty_index)
                                diff -= 1
                    
                        weights.append(t)
                        weight_bins.append(get_weight_bins(self.n, self.bin_dim, G))
			properties_new.append(properties[index])
                        candidate_edges.append(list(G.edges_iter(data='weight')))
                        #print "Debug penalty edges", penalty_edges
		        list_penalty.append(penalty)
		        penalty_edges.extend(list(G.edges()))
                        pos_edges.append(penalty_edges)
                        G_comp = nx.complement(G) 
                        comp_edges = list(G_comp.edges())
                        neg_indices = np.random.choice(range(len(comp_edges)), hparams.neg_sample_size, replace=False)
                        neg_edges_to_be_extended = [comp_edges[index] for index in neg_indices]
                        neg_edges.append(neg_edges_to_be_extended)
                
                    #print("Debug shapes pos_edge", pos_edge)
                    feed_dict.update({self.trajectories: weights})
                    feed_dict.update({self.properties:properties_new})
                    feed_dict.update({self.neg_edges: neg_edges})
                    feed_dict.update({self.edges:np.array(pos_edges)})
                    feed_dict.update({self.n_edges:n_edges})
                    feed_dict.update({self.features: features})
                    feed_dict.update({self.penalty: list_penalty})
                    
                    feed_dict.update({self.weight_bin: weight_bins})
                    
                    _, grad,  train_loss, ll_loss = self.sess.run([self.apply_transform_op,  self.grad, self.loss, self.ll_loss], feed_dict=feed_dict)
                    
                    print("Time size of graph", len(tf.get_default_graph().get_operations()))
                    properties_original = [1.0 - x for x in properties_new]
                    total_train_loss += train_loss 
                    total_ll_loss += ll_loss
                    
                    print("LOSS ",count, train_loss, ll_loss,  properties_original)
                    print("candiadte1", candidate_edges[0])
		    print("candiadte2", candidate_edges[1])
		    print("candidate3", candidate_edges[2])
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
        

    def getembeddings(self, hparams, placeholders, adj, deg, weight_bin, weight):

        eps = np.random.randn(self.n, self.z_dim, 1)

        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d,
                                        hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj})
        feed_dict.update({self.features: deg})
        feed_dict.update({self.input_data: np.zeros([self.k, self.n, self.d])})
        feed_dict.update({self.eps: eps})
        feed_dict.update({self.weight_bin: weight_bin})
        feed_dict.update({self.weight: weight})

        prob, ll, kl, w_edge, embedding = self.sess.run([self.prob, self.ll, self.kl, self.w_edge, self.z_encoded],
                                                        feed_dict=feed_dict)
        return embedding

    def get_masked_candidate_with_atom_ratio_new(self, prob, w_edge, atom_count, num_edges, hde):
        rest = range(self.n)
        nodes = []
        hn = []
        on = []
        nn = []
        cn = []

        for i in range(self.n):
            if atom_count[i] == 1:
                hn.append(i)
            if atom_count[i] == 2:
                on.append(i)
            if atom_count[i] == 3 or atom_count[i] == 5:
                nn.append(i)
            if atom_count[i] == 4:
                cn.append(i)


        nodes.extend(hn)
        nodes.extend(cn)
        nodes.extend(on)
        nodes.extend(nn)

        node_list = atom_count
        print("Debug nodelist", node_list)
        
        indicator = np.ones([self.n, self.bin_dim])
        edge_mask = np.ones([self.n, self.n])
        degree = np.zeros(self.n)

        for node in hn:
            indicator[node][1] = 0
            indicator[node][2] = 0
        for node in on:
            indicator[node][2] = 0

        # two hydrogen atom cannot have an edge between them
        for n1 in hn:
            for n2 in hn:
                edge_mask[n1][n2] = 0
        candidate_edges = []
        # first generate edges joining with Hydrogen atoms sequentially
        index = 0
        i = 0
        hydro_sat = np.zeros(self.n)
        #first handle hydro
        try:
         for node in nodes:
            deg_req = node_list[node]
            d = degree[node]
            list_edges = get_candidate_neighbor_edges(node, self.n)
            if node in hn:
                for i1 in range(self.n):
                    if hydro_sat[i1] == node_list[i1] - 1:
                        edge_mask[i1][node] = 0
                        edge_mask[node][i1] = 0
            while d < deg_req:
                p = normalise_h1(prob, w_edge,  self.bin_dim, indicator, edge_mask, node)
                
                candidate_edges.extend([list_edges[k] for k in
                               np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])

                (u, v, w) = candidate_edges[i]
                degree[u]+= w
                degree[v]+= w
                d += w
                if u in hn:
                    hydro_sat[v] += 1
                if v in hn:
                    hydro_sat[u] += 1
                edge_mask[u][v] = 0
                edge_mask[v][u] = 0
                
                if (node_list[u] - degree[u]) == 0 :
                    indicator[u][0] = 0
                if (node_list[u] - degree[u]) <= 1 :
                    indicator[u][1] = 0
                if (node_list[u] - degree[u]) <= 2:
                    indicator[u][2] = 0

                if (node_list[v] - degree[v]) == 0 :
                    indicator[v][0] = 0
                if (node_list[v] - degree[v]) <= 1 :
                    indicator[v][1] = 0
                if (node_list[v] - degree[v]) <= 2:
                    indicator[v][2] = 0
                
                i+=1 
                #print("Debug candidate_edges", candidate_edges[i - 1])
                #    print("change state", el, degree[el], node_list[el], indicator[el])
                #'''
        except:
         if len(candidate_edges) < 1:
                candidate_edges = []
        candidate_edges_new = []
        for (u, v, w) in candidate_edges:
            if u < v:
                candidate_edges_new.append(str(u) + ' ' + str(v) + ' ' + "{'weight':"+str(w)+"}")
            else:
                candidate_edges_new.append(str(v) + ' ' + str(u) + ' ' + "{'weight':"+str(w)+"}")
        print("Candidate_edges_new", candidate_edges_new)
        return candidate_edges_new


    def get_unmasked_candidate(self, list_edges, prob, w_edge, num_edges):
        # sample 1000 times
        count = 0
        structure_list = defaultdict(int)

        # while (count < 1000):
        while (count < 50):
            indicator = np.ones([self.n, self.bin_dim])
            p, list_edges, w = normalise(prob, w_edge, self.n, self.bin_dim, [], list_edges, indicator)
            candidate_edges = [list_edges[k] for k in
                               np.random.choice(range(len(list_edges)), [num_edges], p=p, replace=False)]
            structure_list[' '.join([str(u) + '-' + str(v) + '-' + str(w) for (u, v, w) in
                                     sorted(candidate_edges, key=itemgetter(0))])] += 1

            # structure_list[sorted(candidate_edges, key=itemgetter(1))] += 1
            count += 1

        # return the element which has been sampled maximum time
        return max(structure_list.iteritems(), key=itemgetter(1))[0]


    def getatoms(self, node, label):
        label_new = np.reshape(label, (node, self.d))
        #print("Debug label original shape:", label_new)
        temp = np.zeros((node, self.d))
        temp.fill(50)
        #print temp, label_new.shape
        minval = np.minimum(label_new, temp)
        label_new = np.exp(minval)
        #print("Debug label exp shape:", label_new)
        s = label_new.shape[0]
        #print("Debug label shape:", label_new.shape, s)

        label_new_sum = np.reshape(np.sum(label_new, axis=1), (s, 1))
        #print("Debug label sum:", label_new_sum.shape, label_new_sum)

        prob_label = label_new / label_new_sum
        
        count = 500
        while(count > 0):
            pred_label = [] 
            #np.zeros(4)
            valency_arr = np.zeros(node)
	    h_c = 0 
	    o_c = 0
	    n_c = 0
	    c_c = 0
            for i in range(node):
                valency = np.random.choice(range(4), [1], p=prob_label[i])
                temp = np.zeros(4)
                temp[valency] += 1
                pred_label.append(temp)
                valency_arr[i] = valency + 1
            	if valency == 0:
			h_c += 1
		if valency == 1:
			o_c += 1
		if valency == 2:
			n_c += 1
		if valency == 3:
			c_c +=1
	    if sum(valency_arr) >= 2 * (self.n - 1):
                	break
            count -= 1
        if sum(valency_arr) < 2 * (self.n -1):
            valency_arr = []
        return (pred_label, valency_arr)


