from utils import *
from config import SAVE_DIR, VAEGConfig
from cell import VAEGCell
from math import log
import tensorflow as tf
import numpy as np
import logging
import copy
import os

from collections import defaultdict
from operator import itemgetter

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
        self.z_dim = hparams.z_dim
        self.bin_dim = hparams.bin_dim
        self.edges = edges
        self.count = 0
        self.mask_weight = hparams.mask_weight

        # For masking we calculated the likelihood like this
        def masked_ll(weight_temp, weight_negative, posscore, posweightscore, temp_pos_score, temp ):
                
                degree = np.zeros([self.n], dtype=np.float32)
                indicator = np.ones([self.n, self.bin_dim], dtype=np.float32)
                ll = 0.0
                for (u, v, w) in self.edges[self.count]:
                    degree[u] += w
                    degree[v] += w
                    
                    modified_weight = tf.reduce_sum(tf.multiply(np.multiply(indicator[u],indicator[v]), weight_temp[u][v])) / weight_negative[u][v]
                    modified_posscore_weighted = modified_weight * posscore[u][v] * 1.0  

                    currentscore = modified_posscore_weighted * 1.0 / (temp_pos_score + temp) 
                    ll += tf.log(currentscore + 1e-9)
                    
                    modified_weight = tf.reduce_sum(tf.multiply(np.multiply(indicator[v],indicator[u]), weight_temp[v][u])) / weight_negative[v][u]
                    modified_posscore_weighted = modified_weight * posscore[v][u] * 1.0  

                    currentscore = modified_posscore_weighted * 1.0 / (temp_pos_score + temp) 
                    ll += tf.log(currentscore + 1e-9)

                    #indicator = np.ones([3], dtype = np.float32)    
                    if degree[u] >=5 :
                        indicator[u][0] = 0
                    if degree[u] >=4  :
                        indicator[u][1] = 0
                    if degree[u] >=3  :
                        indicator[u][2] = 0
                        
                    if degree[v] >=5 :
                        indicator[v][0] = 0
                    if degree[v] >=4  :
                        indicator[v][1] = 0
                    if degree[v] >=3  :
                        indicator[v][2] = 0
                    #If we don't want negative sampling we can uncomment the following
                    '''
                    for i in range(self.n): 
                        modified_weight = tf.reduce_sum(tf.multiply(indicator[u], weight_temp[u][i])) / weight_negative[u][i]
                        modified_posscore_weighted = modified_weight * posscore[u][i] * 1.0  
                        temp_pos_score = temp_pos_score - posweightscore[u][i] + modified_posscore_weighted
                        #posweightscore[u][i] = modified_posscore_weighted
                        #temp_posscore[u][i] = tf.reduce_sum(-posweightscore[u][i] + modified_posscore_weighted)

                        modified_weight = tf.reduce_sum(tf.multiply(indicator[u], weight_temp[i][u])) / weight_negative[i][u]
                        modified_posscore_weighted = modified_weight * posscore[i][u] * 1.0  
                        temp_pos_score = temp_pos_score - posweightscore[i][u] + modified_posscore_weighted
                        #posweightscore[i][u] = modified_posscore_weighted
                        #temp_posscore[i][u] = tf.reduce_sum(-posweightscore[i][u] + modified_posscore_weighted)

                        modified_weight = tf.reduce_sum(tf.multiply(indicator[v], weight_temp[v][i])) / weight_negative[v][i]
                        modified_posscore_weighted = modified_weight * posscore[v][i] * 1.0  
                        temp_pos_score = temp_pos_score - posweightscore[v][i] + modified_posscore_weighted
                        #posweightscore[v][i] = modified_posscore_weighted
                        #temp_posscore[v][i] = tf.reduce_sum(-posweightscore[v][i] + modified_posscore_weighted)

                        modified_weight = tf.reduce_sum(tf.multiply(indicator[v], weight_temp[i][v])) / weight_negative[i][v]
                        modified_posscore_weighted = modified_weight * posscore[i][v] * 1.0  
                        temp_pos_score = temp_pos_score - posweightscore[i][v] + modified_posscore_weighted
                    '''
                return ll
                
        def neg_loglikelihood(prob_dict, w_edge):
            '''
            negative loglikelihood of the edges
            '''
            ll = 0
            k = 0
            with tf.variable_scope('NLL'):
                dec_mat_temp = tf.reshape(prob_dict, [self.n, self.n])
                w_edge_new = tf.reshape(w_edge, [self.n, self.n, self.bin_dim])
            
                #dec_mat = tf.exp(tf.minimum(tf.reshape(prob_dict, [self.n, self.n]),tf.fill([self.n, self.n], 10.0)))
                weight_negative = []
                weight_stack = []

                w_edge_new = tf.exp(tf.minimum(w_edge_new, tf.fill([self.n, self.n, self.bin_dim], 10.0)))
                weight_temp = tf.multiply(self.weight_bin, w_edge_new)
                
                for i in range(self.n):
                    for j in range(self.n):
                        weight_negative.append(tf.reduce_sum(w_edge_new[i][j]))
                        weight_stack.append(tf.reduce_sum(weight_temp[i][j]))
                
                weight_stack = tf.reshape(weight_stack, [self.n, self.n])
                weight_negative = tf.reshape(weight_negative, [self.n, self.n])
    
                w_score = tf.truediv(weight_stack, weight_negative)
                weight_comp = tf.subtract(tf.fill([self.n, self.n], 1.0), w_score)
                
                dec_mat = tf.exp(tf.minimum(dec_mat_temp, tf.fill([self.n, self.n], 10.0)))
                dec_mat = tf.Print(dec_mat, [dec_mat], message="my decscore values:")
                
                comp = tf.subtract(tf.ones([self.n, self.n], tf.float32), self.adj)
                comp = tf.Print(comp, [comp], message="my comp values:")
		
                temp = tf.reduce_sum(tf.multiply(comp,dec_mat))
		negscore = tf.multiply(tf.fill([self.n,self.n], temp+1e-9), weight_comp)
                negscore = tf.Print(negscore, [negscore], message="my negscore values:")
                
                posscore = tf.multiply(self.adj, dec_mat)
                posscore = tf.Print(posscore, [posscore], message="my posscore values:")
    
                posweightscore = tf.multiply(posscore, w_score)
                temp_pos_score = tf.reduce_sum(posweightscore)
                posweightscore = tf.Print(posweightscore, [posweightscore], message="my weighted posscore")

                softmax_out = tf.truediv(posweightscore, tf.add(posweightscore, negscore))
                if self.mask_weight:
                    ll = masked_ll(weight_temp, weight_negative, posscore, posweightscore, temp_pos_score, temp)
                
                else:
                    ll= tf.reduce_sum(tf.log(tf.add(tf.multiply(self.adj, softmax_out), tf.fill([self.n,self.n], 1e-9))),1)
            
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
                k = tf.fill([self.n], tf.cast(self.z_dim, tf.float32))


                temp_stack = []

                for i in range(self.n):
                    temp_stack.append(tf.reduce_prod(tf.square(debug_sigma[i])))

                #print "Debug", tf.stack(temp_stack).shape
                third_term = tf.log(tf.add(tf.stack(temp_stack),tf.fill([self.n],1e-09)))

                #print "debug KL", first_term.shape, second_term.shape, k.shape, third_term.shape, sigma_1[0].shape
                #return 0.5 *tf.reduce_sum((
                return 0.5 * tf.add(tf.subtract(tf.add(first_term ,second_term), k), third_term)
        
	def get_lossfunc(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma, dec_out, w_edge):
            kl_loss = kl_gaussian(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma)  # KL_divergence loss
            likelihood_loss = neg_loglikelihood(dec_out, w_edge)  # Cross entropy loss
            self.ll = likelihood_loss
            self.kl = kl_loss
            return tf.reduce_mean(kl_loss + likelihood_loss)


        self.adj = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='adj')
        self.features = tf.placeholder(dtype=tf.float32, shape=[self.n, self.d], name='features')
        self.weight = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name="weight")
        self.weight_bin = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n, hparams.bin_dim], name="weight_bin")
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.k, self.n, self.d], name='input')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[self.n, self.z_dim, 1], name='eps')

	self.cell = VAEGCell(self.adj, self.weight, self.features, self.z_dim, self.bin_dim)
        self.c_x, enc_mu, enc_sigma, debug_sigma,dec_out, prior_mu, prior_sigma, z_encoded, w_edge= self.cell.call(self.input_data, self.n, self.d, self.k, self.eps, hparams.sample)
	self.prob = dec_out
        #print('Debug', dec_out.shape)
        self.z_encoded = z_encoded
        self.enc_mu = enc_mu
        self.enc_sigma = enc_sigma
        self.w_edge = w_edge
        self.cost = get_lossfunc(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma, dec_out, w_edge)

        print_vars("trainable_variables")
        # self.lr = tf.Variable(self.lr, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.grad = self.train_op.compute_gradients(self.cost)
        self.grad_placeholder = [(tf.placeholder("float", shape=gr[1].get_shape()), gr[1]) for gr in self.grad]
        #self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.grad]  
        #self.tgv = [self.grad]
        # self.apply_transform_op = self.train_op.apply_gradients(self.grad_placeholder)
        #self.apply_transform_op = self.train_op.apply_gradients(self.capped_gvs)
        self.apply_transform_op = self.train_op.apply_gradients(self.grad)

        #self.lr = tf.Variable(self.lr, trainable=False)
        #self.gradient = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-4).compute_gradients(self.cost)
        #self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-4).minimize(self.cost)
        #self.check_op = tf.add_check_numerics_ops()
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

    def train(self, placeholders, hparams, adj,weight, weight_bin, features):
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
        saver = tf.train.Saver(tf.global_variables())

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)

        for epoch in range(num_epochs):
            for i in range(len(adj)):
                
                self.count = i
                if len(self.edges[self.count]) == 0:
                    continue
                # Learning rate decay
                #self.sess.run(tf.assign(self.lr, self.lr * (self.decay ** epoch)))
                feed_dict = construct_feed_dict(lr, dr, self.k, self.n, self.d, decay, placeholders)
                feed_dict.update({self.adj: adj[i]})
	        #print "Debug", features[i].shape
                
                eps = np.random.randn(self.n, self.z_dim, 1)  
                #tf.random_normal((self.n, 5, 1), 0.0, 1.0, dtype=tf.float32)
                
                feed_dict.update({self.features: features[i]})
                feed_dict.update({self.weight_bin: weight_bin[i]})
                feed_dict.update({self.weight: weight[i]})
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.eps: eps})
                
                grad_vals = self.sess.run([g[0] for g in self.grad], feed_dict=feed_dict)
                for j in xrange(len(self.grad_placeholder)):
                    feed_dict.update({self.grad_placeholder[j][0]: grad_vals[j]})
                input_, train_loss, _, probdict, cx, w_edge= self.sess.run([self.input_data ,self.cost, self.apply_transform_op, self.prob, self.c_x, self.w_edge], feed_dict=feed_dict)

                iteration += 1
                
                if iteration % hparams.log_every == 0 and iteration > 0:
                    print("{}/{}(epoch {}), train_loss = {:.6f}".format(iteration, num_epochs, epoch + 1, train_loss))
		    #print(probdict)
                    checkpoint_path = os.path.join(savedir, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path, global_step=iteration)
                    logger.info("model saved to {}".format(checkpoint_path))
        f1 = open(hparams.out_dir+'/iteration.txt','w')
        f1.write(str(iteration))


    
    def getembeddings(self, hparams, placeholders, adj, deg, weight_bin, weight):
         
        eps = np.random.randn(self.n, self.z_dim, 1)

        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj})
	feed_dict.update({self.features: deg})
        feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
        feed_dict.update({self.eps: eps})
        feed_dict.update({self.weight_bin: weight_bin})
        feed_dict.update({self.weight: weight})
   
        prob, ll, kl, w_edge, embedding = self.sess.run([self.prob, self.ll, self.kl, self.w_edge, self.z_encoded],feed_dict=feed_dict)
        return embedding
    
    def sample_graph_slerp(self, hparams, placeholders, s_num, G_good, G_bad, inter, ratio, index, num=10):
        # Agrs :
        # G_good : embedding of the train graph or good sample
        # G_bad : embedding of the bad graph

            list_edges = []
            for i in range(self.n):
                for j in range(i+1, self.n):
                    #list_edges.append((i,j))
                    list_edges.append((i,j,1))
                    list_edges.append((i,j,2))
                    list_edges.append((i,j,3))
            list_weights = [1, 2, 3]
            
            #for sample in range(s_num):
            new_graph = []
            for i in range(self.n):
            #for i in range(index, index+1):
                node_good = G_good[i]
                node_bad = G_bad[i]
                if i == index:
                 if inter == "lerp":
                    #a = lerp(np.reshape(node_good, -1)[:index], np.reshape(node_bad,-1)[:index], ratio).tolist()
                    #b = np.reshape(node_good, -1)[index:].tolist()
                    #a.extend(b)
                    #new_graph.append(a)
                    #new_graph.append(lerp(np.reshape(node_good, -1)[:index], np.reshape(node_bad,-1)[:index], ratio).tolist().extend(np.reshape(node_good, -1)[index:].tolist()))
                    new_graph.append(lerp(np.reshape(node_good, -1), np.reshape(node_bad,-1), ratio))
                 else:
                    #a = slerp(np.reshape(node_good, -1)[:index], np.reshape(node_bad,-1)[:index], ratio).tolist()
                    #b = np.reshape(node_good, -1)[index:].tolist()
                    #a.extend(b)
                    #print("Debug",list(lerp(np.reshape(node_good, -1)[:index], np.reshape(node_bad,-1)[:index], ratio)), np.reshape(node_good, -1)[index:])
                    #new_graph.append(a)
                    #new_graph.append(slerp(np.reshape(node_good, -1)[:index], np.reshape(node_bad, -1)[:index], ratio).tolist().extend(np.reshape(node_good, -1)[index:].tolist()))
                    new_graph.append(slerp(np.reshape(node_good, -1), np.reshape(node_bad, -1), ratio))
                else:
                 new_graph.append(np.reshape(node_good, -1))
            #print "Debug interpolation", len(new_graph), len(new_graph[0]) 
            eps = np.array(new_graph)
            #print "Debug interpolation", eps.shape
            eps = eps.reshape(eps.shape+(1,))
            #print "Debug interpolation", eps.shape
            hparams.sample = True
            #print "EPS", eps
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
            
            #TODO adj and deg are filler and does not required while sampling. Need to clean this part 
            adj = np.zeros([self.n, self.n])
            deg = np.zeros([self.n, 1], dtype=np.float)
            weight_bin = np.zeros([self.n, self.n, self.bin_dim])
            weight = np.zeros([self.n, self.n])
            feed_dict.update({self.adj: adj})
	    feed_dict.update({self.features: deg})
            feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.eps: eps})
            feed_dict.update({self.weight_bin: weight_bin})
            feed_dict.update({self.weight: weight})
   
            prob, ll, kl, w_edge = self.sess.run([self.prob, self.ll, self.kl, self.w_edge],feed_dict=feed_dict)
            prob = np.reshape(prob,(self.n, self.n))
            w_edge = np.reshape(w_edge,(self.n, self.n, self.bin_dim))

            #adj = np.zeros([self.n, self.n])
            indicator = np.ones([self.n, self.bin_dim])
            #print("debug prob", prob)
            p, list_edges, w_new = normalise(prob, w_edge, self.n, self.bin_dim, [], list_edges, indicator)
            #print("Debug p edge", len(p), len(list_edges), p)

            candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[1], p=p, replace=False)]
            #weightlist = [list_weights[k] for k in np.random.choice(range(len(list_weights)),[1], p=w_new[][], replace=False)]

            probtotal = 1.0
            degree = np.zeros([self.n])
        
            for i in range(hparams.edges - 1):
                    (u,v,w) = candidate_edges[i]
                    #(u,v) = candidate_edges[i]
                    #w = weight_lists[i]
                    degree[u] += w
                    degree[v] += w
                    
                    #wscore_u_v = tf.reduce_sum(tf.multiply(indicator, weight_temp[u][v]))
                    #poscore_weighted_u_v = wscore_u_v * posscore[u][v]
                    #if degree[u] >=5 :
                    #    indicator[u][0] = 0
                    if degree[u] >=4  :
                        indicator[u][0] = 0
                        indicator[u][1] = 0

                    if degree[u] >=3  :
                        indicator[u][1] = 0
                        indicator[u][2] = 0

                    #if degree[v] >=5 :
                    #    indicator[v][0] = 0
                    if degree[v] >=4  :
                        indicator[v][0] = 0
                        indicator[v][1] = 0
                    if degree[v] >=3  :
                        indicator[v][1] = 0        
                        indicator[v][2] = 0
                
                    
                    #print("Debug indicator before", indicator, w_edge[u][v])
                    #w_edge[u][v] = np.exp(w_edge[u][v])/sum(np.exp(w_edge[u][v]))
                    #w_edge[u][v] = np.multiply(indicator, w_edge[u][v])
                    #w_edge[v][u] = np.multiply(indicator, w_edge[v][u])
                    #print("Debug indicator after",indicator, w_edge[u][v])
                    #w_edge[u][v] = np.exp(w_edge[u][v])/sum(np.exp(w_edge[u][v]))
                    
                    p, list_edges, w_new = normalise(prob, w_edge, self.n, self.bin_dim, candidate_edges, list_edges, indicator)
                    candidate_edges.extend([ list_edges[k] for k in np.random.choice(range(len(list_edges)),[1], p=p, replace=False)])
            
            for (u,v, w) in candidate_edges:
                with open(hparams.sample_file+'/inter/'+str(index)+inter+str(s_num)+'.txt', 'a') as f:
                        #f.write(str(u)+'\t'+str(v)+'\n')
                        f.write(str(u)+' '+str(v)+' {\'weight\':'+str(w)+'}\n')

            with open(hparams.z_dir+'/inter/'+str(index)+inter+str(s_num)+'.txt', 'a') as f:
                    for z_i in eps:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

            return new_graph


    def get_stat(self, hparams, placeholders, num=10, outdir=None):
        
        adj, features = load_data(hparams.graph_file, hparams.nodes)

        hparams.sample = True
        eps = np.random.randn(self.n, self.z_dim, 1)
        for i in range(len(adj)):
            ll_total = 0.0
            loss_total = 0.0
            prob_derived = 0.0

            for j in range(10):
                eps = np.random.randn(self.n, self.z_dim, 1) 
                feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
                feed_dict.update({self.adj: adj[i]})
	        feed_dict.update({self.features: features[i]})
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.eps: eps})
                prob, ll, z_encoded, enc_mu, enc_sigma, loss, kl = self.sess.run([self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost, self.kl], feed_dict=feed_dict )
                ll_total+= np.mean(ll)
                loss_total+=np.mean(loss)

                prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
                prob = np.divide(prob, np.sum(prob))

                
                for k in range(self.n):
                    for l in range(k+1, self.n):
                        if adj[i][k][l] == 1:
                            prob_derived += log(prob[k][l])
            

            with open(hparams.sample_file+'/reconstruction_ll.txt', 'a') as f:
                    f.write(str(-np.mean(ll_total)//10)+'\n')
        
                #with open(hparams.graph_file+'/kl.txt', 'a') as f:
                #    f.write(str(-np.mean(kl))+'\n')

            with open(hparams.sample_file+'/elbo.txt', 'a') as f:
                    f.write(str(-np.mean(loss_total)//10)+'\n')
            
            with open(hparams.sample_file+'/prob_derived.txt', 'a') as f:
                    f.write(str(-np.mean(loss_total)//10)+'\n')

    def get_masked_candidate(self, list_edges, prob, w_edge, num_edges):

        list_edges_original = copy.copy(list_edges)

        #sample 1000 times
        count  = 0
        structure_list = defaultdict(int)

        while(count < 1000):
            list_edges = copy.copy(list_edges_original)

            indicator = np.ones([self.n, self.bin_dim])
            p, list_edges, w = normalise(prob, w_edge, self.n, self.bin_dim, [], list_edges, indicator)
            candidate_edges = [list_edges[k] for k in
                               np.random.choice(range(len(list_edges)), [1], p=p, replace=False)]

            degree = np.zeros([self.n])
            for i1 in range(num_edges - 1):
                (u, v, w) = candidate_edges[i1]
                degree[u] += w
                degree[v] += w

                #if degree[u] >= 5:
                #    indicator[u][0] = 0
                if degree[u] >= 4:
                    indicator[u][1] = 0
                    indicator[u][0] = 0
                if degree[u] >= 3:
                    indicator[u][1] = 0
                    indicator[u][2] = 0

                #if degree[v] >= 5:
                #    indicator[v][0] = 0
                if degree[v] >= 4:
                    indicator[v][0] = 0
                    indicator[v][1] = 0
                if degree[v] >= 3:
                    indicator[v][1] = 0
                    indicator[v][2] = 0

                p, list_edges, w = normalise(prob, w_edge, self.n, self.bin_dim, candidate_edges, list_edges, indicator)

                candidate_edges.extend([list_edges[k] for k in
                                       np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])
            structure_list[' '.join([str(u)+ '-'+str(v)+'-'+str(w) for (u,v,w) in sorted(candidate_edges)])] += 1
            count += 1

        #return the element which has been sampled maximum time
        print("Debug structure", structure_list)
        return max(structure_list.iteritems(), key=itemgetter(1))[0]

    def get_unmasked_candidate(self, list_edges, prob, w_edge, num_edges):
        # sample 1000 times
        count = 0
        structure_list = defaultdict(int)

        while (count < 1000):
            indicator = np.ones([self.n, self.bin_dim])
            p, list_edges, w = normalise(prob, w_edge, self.n, self.bin_dim, [], list_edges, indicator)
            candidate_edges = [list_edges[k] for k in
                               np.random.choice(range(len(list_edges)), [num_edges], p=p, replace=False)]
            structure_list[' '.join([str(u)+ '-'+str(v)+'-'+str(w) for (u,v,w) in sorted(candidate_edges,key=itemgetter(0))])] += 1

            #structure_list[sorted(candidate_edges, key=itemgetter(1))] += 1
            count += 1

        # return the element which has been sampled maximum time
        return max(structure_list.iteritems(), key=operator.itemgetter(1))[0]

    def sample_graph_posterior(self, hparams,placeholders, adj, features, weights, weight_bins, s_num, node, num=10, outdir=None):
        list_edges = get_candidate_edges(self.n)
        eps = np.random.randn(self.n, self.z_dim, 1)

        train_mu = []
        train_sigma = []
        hparams.sample = False

        # approach 1
        for i in range(len(adj)):
            #list_edges_new = copy.copy(list_edges)
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d,
                                            hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adj[i]})
            feed_dict.update({self.features: features[i]})
            feed_dict.update({self.weight_bin: weight_bins[i]})
            feed_dict.update({self.weight: weights[i]})
            feed_dict.update({self.input_data: np.zeros([self.k, self.n, self.d])})
            feed_dict.update({self.eps: eps})

            prob, ll, z_encoded, enc_mu, enc_sigma, elbo, w_edge = self.sess.run(
                [self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost, self.w_edge],
                feed_dict=feed_dict)
            # prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
            prob = np.reshape(prob, (self.n, self.n))

            w_edge = np.reshape(w_edge, (self.n, self.n, self.bin_dim))
            #indicator = np.ones([self.n, self.bin_dim])
            #p, list_edges_new, w_new = normalise(prob, w_edge, self.n, hparams.bin_dim, [], list_edges_new, indicator)
            if not hparams.mask_weight:
                candidate_edges = self.get_unmasked_candidate(list_edges, prob, w_edge, hparams.edges)
            else:
                candidate_edges = self.get_masked_candidate(list_edges, prob, w_edge, hparams.edges)

            for (u, v, w) in candidate_edges:
                if (u >= 0 and v >= 0):
                    with open(hparams.sample_file + "approach_1_train" + str(i) + "_" + str(s_num) + '.txt', 'a') as f:
                        # print("Writing", u, v, i, s_num)
                        f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')


    def sample_graph_neighborhood(self, hparams,placeholders, adj, features, weights, weight_bins, s_num, node, ratio, num=10, outdir=None):
        list_edges = get_candidate_edges(self.n)

        eps = load_embeddings(hparams.z_dir+'encoded_input0'+'.txt', hparams.z_dim)
        #eps = np.random.randn(self.n, self.z_dim, 1)

        train_mu = []
        train_sigma = []
        hparams.sample = False

        # approach 1
        for i in range(len(adj)):
            #list_edges_new = copy.copy(list_edges)
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d,
                                            hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adj[i]})
            feed_dict.update({self.features: features[i]})
            feed_dict.update({self.weight_bin: weight_bins[i]})
            feed_dict.update({self.weight: weights[i]})
            feed_dict.update({self.input_data: np.zeros([self.k, self.n, self.d])})
            feed_dict.update({self.eps: eps})
            hparams.sample = False
            prob, ll, z_encoded, enc_mu, enc_sigma, elbo, w_edge = self.sess.run(
                [self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost, self.w_edge],
                feed_dict=feed_dict)

            '''
            with open(hparams.z_dir+'encoded_input'+str(i)+'.txt', 'a') as f:
                for z_i in z_encoded:
                    f.write('['+','.join([str(el[0]) for el in z_i])+']\n')
                f.write("\n")
            '''
            
            hparams.sample = True

            for j in range(self.n):
                z_encoded_neighborhood = copy.copy(z_encoded)
                #print("Debug size", z_encoded.shape, z_encoded[0].shape)
                z_encoded_neighborhood[j] = lerp(z_encoded[j], np.ones(z_encoded[j].shape), ratio)
                with open(hparams.z_dir+'interpolated_input1'+str(i)+'.txt', 'a') as f:
                    for z_i in z_encoded_neighborhood:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

                    f.write("\n")

                feed_dict.update({self.eps:z_encoded_neighborhood})
                
                prob, ll, z_encoded_neighborhood, enc_mu, enc_sigma, elbo, w_edge = self.sess.run(
                [self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost, self.w_edge],
                feed_dict=feed_dict)

                # prob = np.triu(np.reshape(prob,(self.n,self.n)),1)

                prob = np.reshape(prob, (self.n, self.n))

                w_edge = np.reshape(w_edge, (self.n, self.n, self.bin_dim))
                #indicator = np.ones([self.n, self.bin_dim])
                #p, list_edges_new, w_new = normalise(prob, w_edge, self.n, hparams.bin_dim, [], list_edges_new, indicator)
                if not hparams.mask_weight:
                    candidate_edges = self.get_unmasked_candidate(list_edges, prob, w_edge, hparams.edges)
                else:
                    candidate_edges = self.get_masked_candidate(list_edges, prob, w_edge, hparams.edges)

                #for (u, v, w) in candidate_edges:
                for uvw in candidate_edges.split():
                    [u,v,w] = uvw.split("-")
                    u = int(u)
                    v = int(v)
                    w = int(w)
                    if (u >= 0 and v >= 0):
                        with open(hparams.sample_file + "approach_1_node_" + str(j) + "_" + str(s_num) + '.txt', 'a') as f:
                            # print("Writing", u, v, i, s_num)
                            f.write(str(u) + ' ' + str(v) + ' {\'weight\':' + str(w) + '}\n')
                #break

    def sample_graph(self, hparams,placeholders, adj, features, weights, weight_bins, s_num, node, num=10, outdir=None):
        
        '''
        Args :
            num - int
                10
                number of edges to be sampled
            outdir - string
            output dir
        '''
        #TODO: the masking part is redundant and hence should be shifted to a function 
        list_edges = []
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                    #list_edges.append((i,j))
                    list_edges.append((i,j,1))
                    list_edges.append((i,j,2))
                    list_edges.append((i,j,3))
        #list_edges.append((-1, -1, 0))    

        list_weight = [1,2,3]
        #adj, weights, weight_bins, features, edges = load_data(hparams.graph_file, node)

        #self.edges = edges
        eps = np.random.randn(self.n, self.z_dim, 1) 
        
        #'''
        with open(hparams.z_dir+'test_prior_'+str(s_num)+'.txt', 'a') as f:
                for z_i in eps:
                    f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

        #tf.random_normal((self.n, 5, 1), 0.0, 1.0, dtype=tf.float32)
        train_mu = []
        train_sigma = []
        hparams.sample = False
        
        #approach 1
        for i in range(len(adj)):
            list_edges_new = copy.copy(list_edges)
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adj[i]})
	    feed_dict.update({self.features: features[i]})
            feed_dict.update({self.weight_bin: weight_bins[i]})
            feed_dict.update({self.weight: weights[i]})
            feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.eps: eps})

            prob, ll, z_encoded, enc_mu, enc_sigma, elbo, w_edge = self.sess.run([self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost, self.w_edge],feed_dict=feed_dict )
            #prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
            prob = np.reshape(prob, (self.n, self.n))
            
            w_edge = np.reshape(w_edge,(self.n, self.n, hparams.bin_dim))
            indicator = np.ones([self.n, self.bin_dim])
            
            p, list_edges_new, w_new = normalise(prob, w_edge, self.n, hparams.bin_dim, [], list_edges_new, indicator)
            if hparams.mask_weight:
                candidate_edges = [ list_edges[k] for k in np.random.choice(range(len(list_edges)),[num], p=p)]
            else:
                candidate_edges = [ list_edges_new[k] for k in np.random.choice(range(len(list_edges_new)),[1], p=p, replace=False)]
                probtotal = 1.0
                #print("Debug candidate edge",len(candidate_edges),i,s_num )
                #adj_new = np.zeros([self.n, self.n])
                degree = np.zeros([self.n])
                for i1 in range(hparams.edges - 1):
                    (u,v,w) = candidate_edges[i1]
                    degree[u] += w
                    degree[v] += w
                    
                    if degree[u] >=5 :
                        indicator[u][0] = 0
                    if degree[u] >=4  :
                        indicator[u][1] = 0
                    if degree[u] >=3  :
                        indicator[u][2] = 0
                        
                    if degree[v] >=5 :
                        indicator[v][0] = 0
                    if degree[v] >=4  :
                        indicator[v][1] = 0
                    if degree[v] >=3  :
                        indicator[v][2] = 0
                    
                    p, list_edges_new, w_new = normalise(prob, w_edge, self.n, self.bin_dim, candidate_edges, list_edges_new, indicator)
                    
                    candidate_edges.extend([ list_edges_new[k] for k in np.random.choice(range(len(list_edges_new)),[1], p=p, replace=False)])

            for (u, v, w) in candidate_edges:
                weight_p = w_edge[u][v]
                weight_p = np.array(weight_p)/np.array(weight_p).sum()
                #w = [ list_weight[k] for k in np.random.choice(range(len(list_weight)),[1], p=w_new[u][v], replace=False)][0]

                if(u>=0 and v >= 0):
                  with open(hparams.sample_file+"approach_1_train"+str(i)+"_"+str(s_num)+'.txt', 'a') as f:
                    #print("Writing", u, v, i, s_num)
                    f.write(str(u)+' '+str(v)+' {\'weight\':'+str(w)+'}\n')
            #''' 
        #approach 2
        hparams.sample=True
        
        eps = np.random.randn(self.n, self.z_dim, 1) 
        with open(hparams.z_dir+'test_prior_'+str(s_num)+'.txt', 'a') as f:
                    for z_i in eps:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj[0]})
	feed_dict.update({self.features:features[0] })
        feed_dict.update({self.weight_bin: weight_bins[0]})
        feed_dict.update({self.weight: weights[0]})

        feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
        feed_dict.update({self.eps: eps})

        prob, ll, z_encoded, kl, sample_mu, sample_sigma, loss, w_edge = self.sess.run([self.prob, self.ll, self.z_encoded, self.kl, self.enc_mu, self.enc_sigma, self.cost, self.w_edge],feed_dict=feed_dict )
        prob = np.reshape(prob,(self.n, self.n))
        w_edge = np.reshape(w_edge,(self.n, self.n, self.bin_dim))
        #print("Debug", prob)        
        
        indicator = np.ones([self.n, 3])
        p, list_edges, w_new = normalise(prob, w_edge, self.n, self.bin_dim, [], list_edges, indicator)
        #print("Debug p edge", len(p), len(list_edges))
        if not hparams.mask_weight:
            candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[hparams.edges], p=p, replace=False)]
        else:    
            candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[1], p=p, replace=False)]

            probtotal = 1.0
            adj = np.zeros([self.n, self.n])
            deg = np.zeros([self.n, 1])
            degree = np.zeros([self.n])
            #print()      
            #indicator = np.ones([self.n, self.n, 3])
            for i in range(hparams.edges - 1):
                    (u,v,w) = candidate_edges[i]
                    degree[u] += w
                    degree[v] += w
                    
                    #wscore_u_v = tf.reduce_sum(tf.multiply(indicator, weight_temp[u][v]))
                    #poscore_weighted_u_v = wscore_u_v * posscore[u][v]
                
                    #indicator = np.ones([3], dtype = np.float32)    
                    
                    if degree[u] >=5 :
                        #indicator[0] = 0
                        indicator[u][0] = 0
                    if degree[u] >=4  :
                        #indicator[1] = 0
                        indicator[u][1] = 0
                    if degree[u] >=3  :
                        #indicator[2] = 0
                        indicator[u][2] = 0
                        
                    if degree[v] >=5 :
                        #indicator[0] = 0
                        indicator[v][0] = 0
                    if degree[v] >=4  :
                        #indicator[1] = 0
                        indicator[v][1] = 0
                    if degree[v] >=3  :
                        #indicator[2] = 0
                        indicator[v][2] = 0

                    #w_edge[u][v] = np.multiply(indicator, w_edge[u][v])
                    
                    p, list_edges, w_new = normalise(prob, w_edge, self.n, self.bin_dim, candidate_edges, list_edges, indicator)
                    #print("Debug p", p, list_edges)
                    candidate_edges.extend([ list_edges[k] for k in np.random.choice(range(len(list_edges)),[1], p=p, replace=False)])

        for (u,v, w) in candidate_edges:
            #w = [ list_weight[i] for i in np.random.choice(range(len(list_weight)),[1], p=w_new[u][v], replace=False)][0]
            with open(hparams.sample_file+"approach_2"+"_"+str(s_num)+'.txt', 'a') as f:
                f.write(str(u)+' '+str(v)+' {\'weight\':'+str(w)+'}\n')
        #ll1 = log(probtotal)
