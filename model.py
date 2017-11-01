from utils import create_dir, pickle_save
from config import SAVE_DIR, VAEGConfig
from datetime import datetime
from ops import print_vars
from cell import VAEGCell

import tensorflow as tf
import numpy as np
import logging
import pickle
import os


class VAEG(VAEGConfig):
    def __init__(self, placeholders, num_nodes, num_features, istest=False):
        self.adj = placeholders['adj']
        self.features = placeholders['features']
        self.features_dim = num_features
        self.input_dim = num_nodes
        self.dropout = placeholders['dropout']
        #logger.info("Building model starts...")
        def crossentropyloss(adj_pred, adj_original):
            '''
            crossentropy loss of the predicted adjacency and the original adjacency matrix
            '''
            pos_weight = float(adj_pred.shape[0] * adj_pred.shape[0] - adj_pred.sum()) / adj_pred.sum()
            norm = adj_pred.shape[0] * adj_pred.shape[0] / float(
                (adj_pred.shape[0] * adj_pred.shape[0] - adj_pred.sum()) * 2)
            with tf.variable_scope('CR_EN'):
                loss = norm * tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(logits=adj_pred, targets=adj_original,
                                                             pos_weight=pos_weight))

            return loss

        def kl_gaussian(mu_1, sigma_1, mu_2, sigma_2):
            '''
                Kullback leibler divergence for two gaussian distributions
            '''
            with tf.variable_scope("kl_gaussisan"):
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
                    - 2 * tf.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
                    + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9, (tf.square(sigma_2))) - 1), 1)

        def get_lossfunc(enc_mu, enc_sigma, prior_mu, prior_sigma, y, x):
            kl_loss = kl_gaussian(enc_mu, enc_sigma, prior_mu, prior_sigma)  # KL_divergence loss
            likelihood_loss = crossentropyloss(y, x)  # Cross entropy loss
            return tf.reduce_mean(kl_loss + likelihood_loss)

        logger.info("Building VAEGCell starts...")
        self.cell = VAEGCell(self.chunk_samples, self.rnn_size, self.latent_size)
        logger.info("Building VAEGCell done.")

        # The adjacency matrix
        self.input_connectivity = tf.placeholder(dtype=tf.float32,
                                         shape=[self.batch_size, self.seq_length, 2 * self.chunk_samples],
                                         name='input_connectivity')
        # The feature vector
        self.input_features = tf.placeholder(dtype=tf.float32,
                                         shape=[self.batch_size, self.seq_length, 2 * self.chunk_samples],
                                         name='input_features')
        # Output or predicted vector[batch_size, seq_length, chunk_samples*2]
        self.output_data = tf.placeholder(dtype=tf.float32,
                                          shape=[self.batch_size, self.seq_length, 2 * self.chunk_samples],
                                          name='target_data')
        # [batch_size, rnn_size]
        self.initial_state_c, self.initial_state_h = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # [seq_length, batch_size, 2*chunk_samples]
            inputs = tf.reshape(inputs, [-1, 2 * self.chunk_samples])  # [seq_length*batch_size, 2*chunk_samples]
            inputs = tf.split(axis=0, num_or_size_splits=self.seq_length,
                              value=inputs)  # seq_length * [batch_size, 2*chunk_samples]

        # [batch_size* seq_length, chunk_samples*2]
        self.target = tf.reshape(self.target_data, [-1, 2 * self.chunk_samples])

        outputs, last_state = tf.contrib.rnn.static_rnn(self.cell, inputs,
                                                        initial_state=(self.initial_state_c, self.initial_state_h))
        # outputs seq_length*tuple*[batch_size, chunk_samples]
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "prior_mu", "prior_sigma"]

        for n, name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])  # [seq_length, batch_size, chunk_samples]
                x = tf.transpose(x, [1, 0, 2])  # [batch_size, seq_length, chunk_samples]
                x = tf.reshape(x, [self.batch_size * self.seq_length, -1])  # [batch_size x seq_length, chunk_samples]
                outputs_reshape.append(x)
        # tuple*[batch_size x seq_length, chunk_samples]
        enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma = outputs_reshape
        self.mu = dec_mu
        self.sigma = dec_sigma

        self.final_state_c, self.final_state_h = last_state
        self.cost = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma, self.target)

        print_vars("trainable_variables")
        self.lr = tf.Variable(self.lr, trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        logger.info("Building model done.")

        self.sess = tf.Session()

    def next_batch(self):
        '''
        3D signal
            [batch_axis, time_axis, chunk_axis]
        = common noise + noise + sin(time_axis[:] + time_offset)

        half of the chunk_axis are all zeros

        Return:
            x, y
            x - 3D ndarray
                [self.batch_size, self.seq_length, 2*self.chunk_samples]
            y - 3D ndarray
                [self.batch_size, self.seq_length, 2*self.chunk_samples]

        '''
        t_offset = np.random.randn(self.batch_size, 1, (2 * self.chunk_samples))
        mixed_noise = np.random.randn(self.batch_size, self.seq_length, (2 * self.chunk_samples)) * 0.01

        x = np.random.randn(self.batch_size, self.seq_length, (2 * self.chunk_samples)) * 0.1 + mixed_noise + np.sin(
            2 * np.pi * (np.arange(self.seq_length)[np.newaxis, :, np.newaxis] / 10. + t_offset))
        y = np.random.randn(self.batch_size, self.seq_length, (2 * self.chunk_samples)) * 0.1 + mixed_noise + np.sin(
            2 * np.pi * (np.arange(1, self.seq_length + 1)[np.newaxis, :, np.newaxis] / 10. + t_offset))

        y[:, :, self.chunk_samples:] = 0.
        x[:, :, self.chunk_samples:] = 0.
        return x, y

    def initialize(self):
        logger.info("Initialization of parameters")
        self.sess.run(tf.global_variables_initializer())

    def restore(self):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        print("Load the model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def train(self):
        create_dir(SAVE_DIR)
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        saver = tf.train.Saver(tf.global_variables())

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load the model from %s" % ckpt.model_checkpoint_path)

        iteration = 0
        for epoch in range(self.num_epochs):
            # Learning rate decay
            self.sess.run(tf.assign(self.lr, self.lr * (self.decay_rate ** epoch)))

            for batch in range(self.n_batches):
                x, y = self.next_batch()
                feed_dict = {model.input_data: x, model.target_data: y}
                train_loss, _, sigma = self.sess.run([self.cost, self.train_op, self.sigma], feed_dict=feed_dict)

                iteration += 1
                if iteration % self.log_every == 0 and iteration > 0:
                    print("{}/{}(epoch {}), train_loss = {:.6f}, std = {:.3f}".format(iteration,
                                                                                      self.num_epochs * self.n_batches,
                                                                                      epoch + 1,
                                                                                      self.chunk_samples * train_loss,
                                                                                      sigma.mean(axis=0).mean(axis=0)))
                    checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
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