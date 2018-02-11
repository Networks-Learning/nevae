from utils import *
from config import SAVE_DIR, VAEGConfig
from datetime import datetime
#from ops import print_vars
from cell import VAEGCell
from math import log
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
        self.z_dim = hparams.z_dim
        self.count = 0
        self.edges = edges
        self.mask_weight = hparams.mask_weight
        
        #self.edges, self.non_edges = edges, non_edges
        #logger.info("Building model starts...")
        def masked_gen(posscore, negscore):
            indicator = []
            for i in range(self.n):
                indicator.append(tf.ones(self.n))
            temp_posscore = tf.reduce_sum(posscore)
            ll = 0.0
            for (u, v) in self.edges[self.count]:
                print("Debug", posscore[0].shape, indicator[0].shape)
                #tf.multiply(tf.reshape(posscore[u], [1, self.n]), indicator[u])[0][v]
                ll += tf.log(tf.multiply(tf.reshape(posscore[u], [1, self.n]), indicator[u])[0][v] / (temp_posscore + negscore[u][v])+1e-09)
                ll += tf.log(tf.multiply(tf.reshape(posscore[v], [1, self.n]), indicator[v])[0][u] / (temp_posscore + negscore[v][u])+1e-09)

                indicator[u] = np.multiply(tf.subtract(tf.ones([1, self.n]), self.adj[v]), indicator[u])
                indicator[v] = np.multiply(tf.subtract(tf.ones([1, self.n]), self.adj[u]), indicator[v])

                temp_posscore = temp_posscore - tf.reduce_sum(posscore[u])
                temp = tf.multiply(indicator[u], tf.reshape(posscore[u], [self.n]))

                temp_posscore += tf.reduce_sum(temp)
                temp_posscore = temp_posscore - tf.reduce_sum(posscore[v]) + tf.reduce_sum(tf.multiply(indicator[v],posscore[v] ))
                
                temp_posscore = temp_posscore - tf.reduce_sum(tf.transpose(posscore)[u]) + tf.reduce_sum(tf.multiply(indicator[u],tf.transpose(posscore)[u] ))
                temp_posscore = temp_posscore - tf.reduce_sum(tf.transpose(posscore)[v]) + tf.reduce_sum(tf.multiply(indicator[v],tf.transpose(posscore)[v] ))

            return ll

        def neg_loglikelihood(prob_dict):
            '''
            negative loglikelihood of the edges
            '''
            ll = 0
            k = 0
            with tf.variable_scope('NLL'):

                dec_mat_temp = tf.reshape(prob_dict, [self.n, self.n])

                '''
                dec_mat_temp = np.zeros((self.n, self.n))
                for i in range(self.n):
                    for j in range(i+1, self.n):
                        print("Debug", prob_dict[k])
                        dec_mat_temp[i][j] = prob_dict[k][0]
                        dec_mat_temp[j][i] = prob_dict[k][0]
                        k+=1
                #'''

                #dec_mat = tf.exp(tf.minimum(tf.reshape(prob_dict, [self.n, self.n]),tf.fill([self.n, self.n], 10.0)))
                dec_mat = tf.exp(tf.minimum(dec_mat_temp, tf.fill([self.n, self.n], 10.0)))

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
                if hparams.mask_weight:
                    ll = masked_gen(posscore, negscore)
                    #ll = masked_ll(posscore, negscore)
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
            self.kl = kl_loss
            return tf.reduce_mean(kl_loss + likelihood_loss)


        self.adj = tf.placeholder(dtype=tf.float32, shape=[self.n, self.n], name='adj')
        self.features = tf.placeholder(dtype=tf.float32, shape=[self.n, self.d], name='features')
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.k, self.n, self.d], name='input')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[self.n, self.z_dim, 1], name='eps')

	self.cell = VAEGCell(self.adj, self.features, self.z_dim)
        self.c_x, enc_mu, enc_sigma, debug_sigma,dec_out, prior_mu, prior_sigma, z_encoded = self.cell.call(self.input_data, self.n, self.d, self.k, self.eps, hparams.sample)
	self.prob = dec_out
        print('Debug', dec_out.shape)
        self.z_encoded = z_encoded
        self.enc_mu = enc_mu
        self.enc_sigma = enc_sigma
        self.cost = get_lossfunc(enc_mu, enc_sigma, debug_sigma,prior_mu, prior_sigma, dec_out)

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
        print("Load the model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def train(self, placeholders, hparams, adj, features):
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

        #f = open(hparams.out_dir+"iteration.txt")
        iteration = 10000
        #1000
        for epoch in range(num_epochs):
            for i in range(len(adj)):
                self.count = i
                # Learning rate decay

                #self.sess.run(tf.assign(self.lr, self.lr * (self.decay ** epoch)))
                feed_dict = construct_feed_dict(lr, dr, self.k, self.n, self.d, decay, placeholders)
                feed_dict.update({self.adj: adj[i]})
	        #print "Debug", features[i].shape
                eps = np.random.randn(self.n, self.z_dim, 1)  
                #tf.random_normal((self.n, 5, 1), 0.0, 1.0, dtype=tf.float32)
                feed_dict.update({self.features: features[i]})
                feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
                feed_dict.update({self.eps: eps})
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
            #'''
            for i in range(len(adj)):
                eps = np.random.randn(self.n, hparams.z_dim, 1) 
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
            #'''
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
            #'''
    
    
    
    def sample_graph_slerp(self, hparams, placeholders, s_num, G_good, G_bad, inter,ratio,num=10):
        # Agrs :
        # G_good : embedding of the train graph or good sample
        # G_bad : embedding of the bad graph

            list_edges = []
            for i in range(self.n):
                for j in range(i+1, self.n):
                    list_edges.append((i,j))
 
            #for sample in range(s_num):
            new_graph = []
            for i in range(self.n):
                node_good = G_good[i]
                node_bad = G_bad[i]
                if inter == "lerp":
                    new_graph.append(lerp(np.reshape(node_good, -1), np.reshape(node_bad,-1), ratio))
                else:
                    new_graph.append(slerp(np.reshape(node_good, -1), np.reshape(node_bad, -1), ratio))

            eps = np.array(new_graph)
            eps = eps.reshape(eps.shape+(1,))

            hparams.sample = True
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
            #TODO adj and deg are filler and does not required while sampling. Need to clean this part 
            
            adj = np.zeros([self.n, self.n])
            deg = np.zeros([self.n, 1], dtype=np.float)

            feed_dict.update({self.adj: adj})
	    feed_dict.update({self.features: deg})
            feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.eps: eps})
            
            prob, ll, kl = self.sess.run([self.prob, self.ll, self.kl],feed_dict=feed_dict)
        
            prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
            prob = np.divide(prob, np.sum(prob))

            print("Debug", prob)
            
            problist  = []
            try:
             for i in range(self.n):
                for j in range(i+1, self.n):
                    problist.append(prob[i][j])
             p = np.array(problist)
             # list to numpy conversion can change negligible precision. so it is desirable to further normalise it
             p /= p.sum()
             max_prob = max(p)
             min_prob = min(p)
             diff = min_prob + (max_prob-min_prob)*0.1
             print("Debug max prob", max_prob, p)
             #candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[num], p=p, replace=False)]
             candidate_edges = [list_edges[i] for i in range(len(list_edges)) if p[i] >= diff]
            except:
             return
            #adj = np.zeros([self.n, self.n])
            probmul = 1.0
        
            for (u,v) in candidate_edges:
                #adj[u][v] = 1
                #adj[v][u] = 1
                probmul*= prob[u][v]
                with open(hparams.sample_file+'/inter/'+inter+str(s_num)+'.txt', 'a') as f:
                        f.write(str(u)+'\t'+str(v)+'\n')
            
            with open(hparams.z_dir+'/inter/'+inter+str(s_num)+'.txt', 'a') as f:
                    for z_i in eps:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

            #kl_gaussian_mul(np.mean(G_good, axis=0), np.diag(np.var(G_good, axis=0)), np.mean(G_bad, axis = 0), np.diag(np.var(G_bad, axis = 0)))
            #ll1 = log(probmul)
    
            #with open(hparams.sample_file+'/inter/ll.txt', 'a') as f:
            #    f.write(str(ll1)+'\n')

            #kl1 = np.mean(kl)
            #with open(hparams.sample_file+'/inter/kl.txt', 'a') as f:
            #    f.write(str(kl1)+'\n')
            #G_bad = new_graph
            return new_graph

    def kl_gaussian_mul(self, mu_1, sigma_1,  mu_2, sigma_2):
            '''
                Kullback leibler divergence for two gaussian distributions
            '''
            #print("Debug sigma1", debug_sigma_1, len(debug_sigma_1[0]))
            #print sigma_1.shape, sigma_2.shape
            n = self.n
            temp_stack_1 = []
            temp_stack_2 = []
            #debug_sigma_1 = np.diag(sigma_1)
            #debug_sigma_2 = np.diag(sigma_2)
            for i in range(n):
                    #print("DEBUG i", i)
                    temp_stack_1.append(np.prod(sigma_1[i].diagonal()))
                    temp_stack_2.append(np.prod(sigma_2[i].diagonal()))

            # Inverse of diaginal covariance
            ones = np.ones(sigma_2.shape)
            inverse_sigma_2 = np.subtract(ones, np.true_divide(ones, np.add(ones, sigma_2)))
            #inverse_sigma_2 = tf.matrix_diag(np.true_divide(np.ones(np.shape(debug_sigma_2)), debug_sigma_2))

            term_2 = []
            print "DEBUG2", len(inverse_sigma_2)
            for i in range(n):
                    term_2.append(np.trace(np.matmul(inverse_sigma_2[i], sigma_1[i])))
            # Difference between the mean
            term_3 = []
            k = np.zeros([self.n])
            k.fill(mu_1.shape[1])
            diff_mean = np.subtract(mu_2, mu_1)


                
            for i in range(self.n):
                    term_3.append(np.matmul(np.matmul(np.transpose(diff_mean[i]), inverse_sigma_2[i]), diff_mean[i]))
            
            term1 = np.log(np.true_divide(temp_stack_2, temp_stack_1))
            #term2 = np.trace(term_2[])
            #print "Debug", len(term1), len(term_2), len(term_3), len(term_2), len(term_2[0][0])

            KL = 0.5 * np.subtract(np.add(np.add(term1, term_2) , term_3), k)
            #KL = tf.Print(KL, [KL], message="my KL values:")

            #print("Debug mu1", tf.shape(mu_1)[1])
            return np.sum(KL)

    def get_stat(self, hparams, placeholders, num=10, outdir=None):
        
        adj, features, edges = load_data(hparams.graph_file, hparams.nodes)

        #for i in range(self.n):
        #    deg[i][0] = 2 * np.sum(adj[i])/(self.n*(self.n - 1))
        hparams.sample = True
        eps = np.random.randn(self.n, self.z_dim, 1)
        if hparams.sample:
            print("Debug Sample", hparams.sample)
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
                            prob_derived += log(prob[k][l]+0.1)
            


            #with open(hparams.sample_file+'/reconstruction_ll.txt', 'a') as f:
            with open(hparams.out_dir+'/reconstruction_ll1.txt', 'a') as f:
                    f.write(str(-np.mean(ll_total)//10)+'\n')
        
                #with open(hparams.graph_file+'/kl.txt', 'a') as f:
                #    f.write(str(-np.mean(kl))+'\n')


            #with open(hparams.sample_file+'/elbo.txt', 'a') as f:
            with open(hparams.out_dir+'/elbo1.txt', 'a') as f:
                    f.write(str(-np.mean(loss_total)//10)+'\n')
            
            
            #with open(hparams.sample_file+'/prob_derived.txt', 'a') as f:
            with open(hparams.out_dir+'/prob_derived1.txt', 'a') as f:
                    f.write(str(-np.mean(loss_total)//10)+'\n')

    
    
    def zspace_analysis(self, hparams, placeholders, num=10, outdir=None):
        adj, features = load_data(hparams.graph_file, hparams.nodes)
        eps = np.random.randn(self.n, self.z_dim, 1) 
        train_z = []
        list_edges = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                list_edges.append((i,j))
        for i in range(len(adj)):
            hparams.sample = False
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adj[i]})
	    feed_dict.update({self.features: features[i]})
            feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.eps: eps})

            prob, ll, z_encoded, enc_mu, enc_sigma, elbo = self.sess.run([self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost],feed_dict=feed_dict )
            train_z.append(z_encoded)
            
            with open(hparams.z_dir+'train_'+str(i)+'.txt', 'a') as f:
                        for z_i in z_encoded:
                            f.write('['+','.join([str(el[0]) for el in z_i])+']\n')
            
            prob = np.triu(np.reshape(prob,(self.n,self.n)),1)
            prob = np.divide(prob, np.sum(prob))

            problist  = []
            for k in range(self.n):
                for l in range(k+1, self.n):
                    problist.append(prob[k][l])
            p = np.array(problist)
            p /= p.sum()
            if i<20:
                num = 32
            else:
                num = 78
            candidate_edges = [ list_edges[k] for k in np.random.choice(range(len(list_edges)),[num], p=p)]

            probtotal = 1.0
            adjnew = np.zeros([self.n, self.n])
            featuresnew = np.zeros([self.n, 1])
            
            for (u,v) in candidate_edges:
                probtotal *= prob[u][v]
                adjnew[u][v]=1
                adjnew[v][u] = 1
                featuresnew[u][0]+= 1//self.n
                featuresnew[v][0]+=1//self.n
                if i<20:
                    with open(hparams.sample_file+"type_1_test"+"_"+str(i)+'.txt', 'a') as f:
                        f.write(str(u)+'\t'+str(v)+'\n')
                else:
                    with open(hparams.sample_file+"type_2_test"+"_"+str(i)+'.txt', 'a') as f:
                        f.write(str(u)+'\t'+str(v)+'\n')
            #hparams.sample=False
            eps1 = np.random.randn(self.n, self.z_dim, 1) 
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adjnew})
	    feed_dict.update({self.features: featuresnew})
            feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.eps: eps1})
            prob, z_encoded  = self.sess.run([self.prob, self.z_encoded], feed_dict=feed_dict )
            print("DebugZ", len(z_encoded), len(z_encoded[0]))
            if i < 20:
                with open(hparams.z_dir+'type_1_test_'+str(i)+'.txt', 'a') as f:
                        for z_i in z_encoded:
                            f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

            else:
                with open(hparams.z_dir+'type_2_test_'+str(i)+'.txt', 'a') as f:
                        for z_i in z_encoded:
                            f.write('['+','.join([str(el[0]) for el in z_i])+']\n')
            with open(hparams.sample_file+'ll_'+'.txt', 'a') as f:
                        f.write(str(-np.mean(prob))+'\n')

        # Interpolation Finding the likelihood
        count = 0
        for i in range(20):
            for j in range(20,40):
                self.sample_graph_slerp(hparams, placeholders, count, train_z[i], train_z[j], "slerp", 50)
                count += 1
                self.sample_graph_slerp(hparams, placeholders, count, train_z[i], train_z[j], "lerp", 50)
                count += 1

    def getcandidate(self, num, n, p, prob, list_edges):
        print "Inside gencanidate" 
        adj = np.zeros([n,n])
        candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[1], p=p)]
        indicator = np.ones([n, n])
        unseen = np.ones(n)
        probnew = prob
        for k in range(num-1):
            (u, v) = candidate_edges[k]
            adj[u][v]=1
            adj[v][u]=1
            #unseen[u] = 0
            #unseen[v] = 0
            indicator[u] = np.multiply(np.multiply(np.subtract(np.ones(n), adj[v]), indicator[u]), unseen)
            indicator[v] = np.multiply(np.multiply(np.subtract(np.ones(n), adj[u]), indicator[v]), unseen)
            probnew = np.multiply(np.multiply(probnew,indicator), np.transpose(indicator))
            problist = []
            for i in range(self.n):
                for j in range(i+1, self.n):
                    if (i, j) in candidate_edges:
                        if (i, j) in list_edges:
                            list_edges.remove((i,j))
                        continue
                    problist.append(probnew[i][j])
            p = np.array(problist)
            p /= p.sum()
            print "Debug p", p
            candidate_edges.extend([ list_edges[i] for i in np.random.choice(range(len(list_edges)),[1], p=p)])
        
        return candidate_edges
    def getembeddings(self, hparams, placeholders, adj, deg):
                 
        eps = np.random.randn(self.n, self.z_dim, 1)
        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj})
        feed_dict.update({self.features: deg})
        feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
        feed_dict.update({self.eps: eps})
        prob, ll, kl,  embedding = self.sess.run([self.prob, self.ll, self.kl, self.z_encoded],feed_dict=feed_dict)
        return embedding
    
    def sample_graph(self, hparams, placeholders, s_num, node, num=10, outdir=None, eps_passed=None):
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
        adj, features, edges = load_data(hparams.graph_file, node)
        eps = np.random.randn(self.n, self.z_dim, 1) 
        with open(hparams.z_dir+'test_prior_'+str(s_num)+'.txt', 'a') as f:
                    for z_i in eps:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

        #tf.random_normal((self.n, 5, 1), 0.0, 1.0, dtype=tf.float32)
        train_mu = []
        train_sigma = []
        hparams.sample = False
        for i in range(len(adj)):
            feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
            feed_dict.update({self.adj: adj[i]})
	    feed_dict.update({self.features: features[i]})
            feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
            feed_dict.update({self.eps: eps})

            prob, ll, z_encoded, enc_mu, enc_sigma, elbo = self.sess.run([self.prob, self.ll, self.z_encoded, self.enc_mu, self.enc_sigma, self.cost],feed_dict=feed_dict )
            

            prob = np.triu(np.exp(np.reshape(prob,[self.n,self.n])),1)
            prob = np.divide(prob, np.sum(prob))

            problist  = []
            for i in range(self.n):
                for j in range(i+1, self.n):
                    problist.append(prob[i][j])
            p = np.array(problist)
            p /= p.sum()
        
            if hparams.mask_weight:
                candidate_edges = self.getcandidate(num, self.n, p, prob, list_edges)
            else:
                candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[num], p=p)]

            probtotal = 1.0

            for (u,v) in candidate_edges:
                probtotal *= prob[u][v]
                with open(hparams.sample_file+"approach_1_train"+str(i)+"_"+str(s_num)+'.txt', 'a') as f:
                        f.write(str(u)+' '+str(v)+' {}'+'\n')

            #ll1 = np.mean(ll)
            ll1 = log(probtotal)
            with open(hparams.sample_file+"/approach_1_train"+str(i)+'_ll.txt', 'a') as f:
                f.write(str(ll1)+"\t"+str(np.mean(ll))+"\t"+str(np.mean(elbo))+'\n')

        
        #approach 2
        hparams.sample=True
        
        eps = np.random.randn(self.n, self.z_dim, 1) 
        
        if eps_passed != None:
            eps = eps_passed

        with open(hparams.z_dir+'test_prior_'+str(s_num)+'.txt', 'a') as f:
                    for z_i in eps:
                        f.write('['+','.join([str(el[0]) for el in z_i])+']\n')

        feed_dict = construct_feed_dict(hparams.learning_rate, hparams.dropout_rate, self.k, self.n, self.d, hparams.decay_rate, placeholders)
        feed_dict.update({self.adj: adj[0]})
	feed_dict.update({self.features:features[0] })
        feed_dict.update({self.input_data: np.zeros([self.k,self.n,self.d])})
        feed_dict.update({self.eps: eps})

        prob, ll, z_encoded, kl, sample_mu, sample_sigma, loss = self.sess.run([self.prob, self.ll, self.z_encoded, self.kl, self.enc_mu, self.enc_sigma, self.cost],feed_dict=feed_dict )
        
        prob = np.triu(np.exp((np.reshape(prob,(self.n,self.n)))),1)
        prob = np.divide(prob, np.sum(prob))

        problist  = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                problist.append(prob[i][j])
        p = np.array(problist)
        p /= p.sum()
        if hparams.mask_weight:
                candidate_edges = self.getcandidate(num, self.n, p, prob, list_edges)
        else:
                candidate_edges = [ list_edges[i] for i in np.random.choice(range(len(list_edges)),[num], p=p, replace=False)]


        probtotal = 1.0
        adj = np.zeros([self.n, self.n])
        deg = np.zeros([self.n, 1])

        for (u,v) in candidate_edges:
            #adj[u][v] += 1
            #adj[v][u] += 1
            probtotal *= prob[u][v]
            with open(hparams.sample_file+"approach_2"+"_"+str(s_num)+'.txt', 'a') as f:
                    f.write(str(u)+' '+str(v)+' {}'+'\n')
        ll1 = log(probtotal)
        
        with open(hparams.sample_file+'/reconstruction_ll.txt', 'a') as f:
            f.write(str(np.mean(ll))+'\n')

        with open(hparams.sample_file+'/elbo.txt', 'a') as f:
            f.write(str(np.mean(loss))+'\n')

