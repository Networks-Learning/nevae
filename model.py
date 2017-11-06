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

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VAEG(VAEGConfig):
    def __init__(self, hparams, placeholders, num_nodes, num_features, edges, non_edges, istest=False):
        self.features_dim = num_features
        self.input_dim = num_nodes
        self.dropout = placeholders['dropout']
        self.k = hparams.random_walk
        self.lr = placeholders['lr']
        self.n = num_nodes
        self.d = num_features

        self.edges, self.non_edges = edges, non_edges


        #logger.info("Building model starts...")
        def neg_loglikelihood(prob_dict):
            '''
            negative loglikelihood of the edges
            '''
            ll = 0
            with tf.variable_scope('NLL'):
                for (u,v) in self.edges:
                    ll += tf.log(prob_dict[u][v])
            return (-ll)

        def kl_gaussian(mu_1, sigma_1, mu_2, sigma_2):
            '''
                Kullback leibler divergence for two gaussian distributions
            '''
            with tf.variable_scope("kl_gaussisan"):
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
                    - 2 * tf.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
                    + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9, (tf.square(sigma_2))) - 1), 1)

        def get_lossfunc(enc_mu, enc_sigma, prior_mu, prior_sigma, dec_out):
            kl_loss = kl_gaussian(enc_mu, enc_sigma, prior_mu, prior_sigma)  # KL_divergence loss
            likelihood_loss = neg_loglikelihood(dec_out)  # Cross entropy loss
            return tf.reduce_mean(kl_loss + likelihood_loss)

        self.cell = VAEGCell(self.adj, self.features, self.edges, self.non_edges)

        self.adj = tf.placeholder(dtype=tf.int32, shape=[self.n, self.n], name='adj')
        self.features = tf.placeholder(dtype=tf.float32, shape=[self.n, self.d], name='features')
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.k, self.n, self.d], name='input')

        enc_mu, enc_sigma, dec_out, prior_mu, prior_sigma = self.cell.call(self.input_data, self.n, self.d, self.k)
        self.prob = dec_out
        self.cost = get_lossfunc(enc_mu, enc_sigma, prior_mu, prior_sigma, dec_out)

        print_vars("trainable_variables")
        self.lr = tf.Variable(self.lr, trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        self.sess = tf.Session()

    def initialize(self):
        logger.info("Initialization of parameters")
        self.sess.run(tf.global_variables_initializer())

    def restore(self, savedir):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(savedir)
        print("Load the model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def train(self,placeholders, hparams):
        savedir = hparams.out_dir
        lr = hparams.learning_rate
        dr = hparams.dropout_rate

        # training
        num_epochs = hparams.num_epochs
        create_dir(savedir)
        ckpt = tf.train.get_checkpoint_state(savedir)
        saver = tf.train.Saver(tf.global_variables())

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)

        iteration = 0
        #for i in range(k):
        for epoch in range(num_epochs):
                # Learning rate decay
                self.sess.run(tf.assign(lr, lr * (dr ** epoch)))

                feed_dict = construct_feed_dict(self.adj, self.features, lr, dr, self.k, self.n, self.d, placeholders)
                #feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                #outs = self.sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
                #train_loss, _, _= self.sess.run([self.cost, self.train_op], feed_dict)
                train_loss, _ = self.sess.run([self.cost, self.train_op], feed_dict=feed_dict)

                iteration += 1
                if iteration % self.log_every == 0 and iteration > 0:
                    print("{}/{}(epoch {}), train_loss = {:.6f}".format(iteration, num_epochs, epoch + 1, train_loss))
                    checkpoint_path = os.path.join(savedir, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path, global_step=iteration)
                    logger.info("model saved to {}".format(checkpoint_path))

    def sample(self, num=4410, start=None):
        '''
        Args :
            num - int
                4410
            start - sequence
                None => generate [1, 1, 2*self.chunk_samples]
                start.shape==1 => generate [1, 1, 2*self.chunk_samples]
                start.shape==2 [seq, 2*self.chunk_samples]
                => generate(
        Return :
            chunks -
            mus -
            sigmas -
        '''

        def sample_gaussian(mu, sigma):
            return mu + (sigma * np.random.randn(*sigma.shape))

        # Initial condition
        prev_state = self.sess.run(self.cell.zero_state(1, tf.float32))  # [batch_size, rnn_size]

        if start is None:
            prev_x = np.random.randn(1, 1, 2 * self.chunk_samples)
        elif len(start.shape) == 1:
            prev_x = start[np.newaxis, np.newaxis, :]
        elif len(start.shape) == 2:
            for i in range(start.shape[0] - 1):
                prev_x = start[i, :]  # [2*self.chunk_samples]
                prev_x = prev_x[np.newaxis, np.newaxis, :]  # [1, 1, 2*self.chunk_samples]

                feed_dict = {
                    self.input_data: prev_x,
                    self.initial_state_c: prev_state[0],
                    self.initial_state_h: prev_state[1]
                }

                [prev_state_c, prev_state_h] = self.sess.run(
                    [self.mu, self.sigma, self.final_state_c, self.final_state_h],
                    feed_dict=feed_dict
                )
                prev_state = prev_state_c, prev_state_h

            prev_x = start[-1, :]  # [2*self.chunk_samples]
            prev_x = prev_x[np.newaxis, np.newaxis, :]  # [1,1,2*self.chunk_samples]

        chunks = np.zeros((num, 2 * self.chunk_samples), dtype=np.float32)
        mus = np.zeros((num, self.chunk_samples), dtype=np.float32)
        sigmas = np.zeros((num, self.chunk_samples), dtype=np.float32)

        for i in range(num):
            feed_dict = {
                self.input_data: prev_x,
                self.initial_state_c: prev_state[0],
                self.initial_state_h: prev_state[1]
            }

            [o_mu, o_sigma, next_state_c, next_state_h] = self.sess.run(
                [self.mu, self.sigma, self.final_state_c, self.final_state_h],
                feed_dict=feed_dict
            )
            next_x = np.hstack(
                (
                    sample_gaussian(o_mu, o_sigma), np.zeros((1, self.chunk_samples))
                )
            )  # [1, 2*self.chunk_samples]
            chunks[i] = next_x
            mus[i] = o_mu
            sigmas[i] = o_sigma

            prev_x = np.zeros((1, 1, 2 * self.chunk_samples), dtype=np.float32)
            prev_x[0] = next_x
            prev_state = next_state_c, next_state_h

        return chunks, mus, sigmas
