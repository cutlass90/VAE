import random
import os

import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm


from model_abstract import Model

class VAE(Model):
    """A class representing Variational Autoencoder"""

    def __init__(self, input_dim, z_dim, do_train, scope='VAE'):

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.do_train = do_train
        self.scope = scope

        self.activation = tf.nn.relu
        self.summary = []

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.variable_scope(scope):
                self.create_graph()
            if do_train:
                self.cost = self.create_cost_graph(logits=self.logits,
                    original=self.x, z_mu=self.z_mu, z_log_sigma=self.z_log_sigma)
                self.train_step = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self.cost)
                self.train_writer, self.test_writer = self.create_summary_writers('summary/VAE')
                self.merged = tf.summary.merge(self.summary)

            self.sess = self.create_session()
            self.sess.run(tf.global_variables_initializer())
            self.stored_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            self.saver = tf.train.Saver(self.stored_vars, max_to_keep=1000)


    # --------------------------------------------------------------------------
    def create_graph(self):
        print('Create_graph')
        self.x, self.input_z, self. learning_rate, self.sampling, self.is_training =  self._input_graph()

        self.z, self.z_mu, self.z_log_sigma = self.encoder(self.x)
        self.summary.append(tf.summary.histogram('Z hist', self.z))
        self.logits, self.x_ = self.decoder(self.z)
        _, self.rec_x = self.decoder(self.input_z, reuse=True)
        print('Done!')

    # --------------------------------------------------------------------------
    def _input_graph(self):
        print('\t_input_graph')

        x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
        input_z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='input_z')
        learning_rate = tf.placeholder(tf.float32, (), name='learn_rate')
        is_training = tf.placeholder(tf.bool, (), name='is_training')
        sampling = tf.placeholder(tf.bool, (), name='sampling')

        return x, input_z, learning_rate,  sampling, is_training

    # --------------------------------------------------------------------------
    def encoder(self, x, reuse=False):
        print('\tencoder')

        with tf.variable_scope('encoder', reuse=reuse):
            x = tf.reshape(x, (-1, 28, 28, 1))
            regularizer = slim.l2_regularizer(0.001)
            # encoder
            net = slim.conv2d(
                              x,
                              32,
                              [3, 3],
                              activation_fn=self.activation,
                              weights_regularizer=regularizer)
            net = slim.max_pool2d(net, [2, 2], stride=2)
            net = slim.batch_norm(
                                  net,
                                  scale=True,
                                  updates_collections=None,
                                  is_training=self.is_training)
            net = slim.conv2d(
                              net,
                              64,
                              [3, 3],
                              activation_fn=self.activation,
                              weights_regularizer=regularizer)
            net = slim.max_pool2d(net, [2, 2], stride=2)
            net = slim.flatten(net)
            net = slim.fully_connected(
                                        net,
                                        2 * self.z_dim,
                                        activation_fn=None,
                                        weights_regularizer=regularizer)
            #split the layer to mu and sigma
            z_mu, z_log_sigma = tf.split(net, 2, 1)

            z = tf.cond(self.sampling, 
                lambda: self.GaussianSample(z_mu, tf.exp(z_log_sigma)),
                lambda: z_mu)

        return z, z_mu, z_log_sigma

    # --------------------------------------------------------------------------
    def decoder(self, z, reuse=False):
        print('\tdecoder')

        with tf.variable_scope(self.scope):
            with tf.variable_scope('decoder', reuse=reuse):
                regularizer = slim.l2_regularizer(0.001)

                net = slim.fully_connected(
                                           z,
                                           7 * 7 * 64,
                                           activation_fn=self.activation,
                                           weights_regularizer=regularizer)
                net = tf.reshape(net, (-1, 7, 7, 64))
                net = slim.conv2d_transpose(
                                            net,
                                            32,
                                            [3, 3],
                                            stride=2,
                                            activation_fn=self.activation,
                                            weights_regularizer=regularizer)
                net = slim.batch_norm(
                                      net,
                                      scale=True,
                                      updates_collections=None,
                                      is_training=self.is_training)
                net = slim.conv2d_transpose(
                                            net,
                                            1,
                                            [3, 3],
                                            stride=2,
                                            activation_fn=self.activation,
                                            weights_regularizer=regularizer)
                net = slim.flatten(net)
                logits = slim.fully_connected(
                                              net,
                                              self.input_dim,
                                              activation_fn=None,
                                              weights_regularizer=regularizer)

        return logits, tf.nn.sigmoid(logits)

    # --------------------------------------------------------------------------
    def create_cost_graph(self, logits, original, z_mu, z_log_sigma):
        print('\tcreate_cost_graph')
        self.ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=original), 1)
        self.kl_loss = tf.reduce_sum(self.KL(z_mu, tf.exp(z_log_sigma)), 1)
        self.l2_loss = tf.add_n(tf.losses.get_regularization_losses())

        self.summary.append(tf.summary.scalar('Cross entropy loss', tf.reduce_mean(self.ce_loss)))
        self.summary.append(tf.summary.scalar('L2 loss', self.l2_loss))
        self.summary.append(tf.summary.scalar('KL loss', tf.reduce_mean(self.kl_loss)))
        return tf.reduce_mean(self.ce_loss + self.kl_loss, 0) + self.l2_loss


    # --------------------------------------------------------------------------
    def train_model(self, data_loader, batch_size,  learn_rate_start,
        learn_rate_end, n_iter, save_model_every_n_iter, path_to_model):
        print('\n\t----==== Training ====----')
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)
            x, _ = data_loader.train.next_batch(batch_size)
            feed_dict = {
                         self.x: x,
                         self.learning_rate: learn_rate,
                         self.is_training: True,
                         self.sampling:True}
            _, summary = self.sess.run([self.train_step, self.merged],
                feed_dict=feed_dict)
            self.train_writer.add_summary(summary, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')


    # --------------------------------------------------------------------------
    # def predict(self, data_loader):

    #     x = random.sample(list(data_loader.validation.images), 100)
    #     feed_dict = {self.x: x,
    #                  self.is_training: False}
    #     ce, kl = self.sess.run([self.ce_loss, self.kl_loss], feed_dict=feed_dict)        
    #     print('Cross-entropy loss: {}, KL loss: {}'.format(ce.mean(), kl.mean()))

    # --------------------------------------------------------------------------
    def get_z(self, x):
        return self.sess.run(self.z, feed_dict={self.x: x,
                                                self.is_training: False,
                                                self.sampling: False})

    # --------------------------------------------------------------------------
    def reconstruct(self, x):
        return self.sess.run(self.x_, feed_dict={self.x: x,
                                                self.is_training: False,
                                                self.sampling: False}) 

    # --------------------------------------------------------------------------
    def reconstruct_from_z(self, z):
        x = self.sess.run(self.rec_x, feed_dict={self.is_training: False,
                                                self.input_z:z})
        return x        

    # --------------------------------------------------------------------------
    def KL(self, mu, sigma, mu_prior=0.0, sigma_prior=1.0, eps=1e-7):
        return -(1/2)*(1 + tf.log(eps + (sigma/sigma_prior)**2) \
            - (sigma**2 + (mu - mu_prior)**2)/sigma_prior**2)

    # --------------------------------------------------------------------------
    def GaussianSample(self, mu, sigma):
        return mu + sigma*tf.random_normal(tf.shape(mu), dtype=tf.float32)


################################################################################
def test_VAE():
    vae = VAE(input_dim=784, z_dim=2, do_train=True)
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    vae.train_model(data_loader=mnist, batch_size=128, learn_rate_start=1e-3,
        learn_rate_end=1e-4, n_iter=10000, save_model_every_n_iter=99999, path_to_model='models/vae')




if __name__ == '__main__':
    test_VAE()