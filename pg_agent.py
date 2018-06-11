'''
General policy-gradient class
Adapted from https://github.com/awjuliani/DeepRL-Agents/blob/master/Policy-Network.ipynb
Also see Karpathy's blog: http://karpathy.github.io/2016/05/31/rl/
'''

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

class pg_agent():
    def __init__(self, lr, s_size, a_size, h_size):
        ### establish feed-forward NN.  agent takes state and produces action
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        '''
        establish training procedure. feed reward and chosen action into the network
        to compute the loss, and use it to update the network.
        '''
        self.reward_holder = tf.placeholder(shape=[None], dtpye=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indices = tf.range(0, tf.shape(self.output)[0]) + tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indices)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

    def discount_rewards(self, r, gamma = 0.99):
        ''' take 1d array of float rewards and compute discounted reward '''
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
