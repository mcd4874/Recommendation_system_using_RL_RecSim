

import os
import argparse
import numpy as np
import tensorflow as tf
import pprint as pp

from collections import deque
import random
from tensorflow import keras as keras

from tensorflow.keras import backend as K
class Actor(object):
    """policy function approximator"""

    def __init__(self, sess, s_dim, a_dim, batch_size, output_size, weights_len, tau, learning_rate, scope="actor"):
        """

        :param sess:
        :param s_dim: state dimension (# of items * embedding size)
        :param a_dim: action dimension ( # of action * embedding size ) # actions mean how many items recommend to send back
        :param batch_size:
        :param output_size: # actions mean how many items recommend to send back
        :param weights_len: # size of the embedding vector
        :param tau:
        :param learning_rate:
        :param scope:
        """
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.output_size = output_size
        self.weights_len = weights_len
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        # with tf.variable_scope(self.scope):
        #     # estimator actor network
        #     self.state, self.action_weights = self._build_net("estimator_actor")
        #     self.network_params = tf.trainable_variables()
        #     # self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimator_actor')
        #
        #     # target actor network
        #     self.target_state, self.target_action_weights = self._build_net("target_actor")
        #     self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        #     # self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')
        #
        #     # operator for periodically updating target network with estimator network weights
        #     self.update_target_network_params = [
        #         self.target_network_params[i].assign(
        #             tf.multiply(self.network_params[i], self.tau) +
        #             tf.multiply(self.target_network_params[i], 1 - self.tau)
        #         ) for i in range(len(self.target_network_params))
        #     ]
        #     self.hard_update_target_network_params = [
        #         self.target_network_params[i].assign(
        #             self.network_params[i]
        #         ) for i in range(len(self.target_network_params))
        #     ]
        #
        #     self.a_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        #     self.params_gradients = list(
        #         map(
        #             lambda x: tf.div(x, self.batch_size * self.a_dim),
        #             tf.gradients(tf.reshape(self.action_weights, [self.batch_size, self.a_dim]),
        #                          self.network_params, -self.a_gradient)
        #         )
        #     )
        #     self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
        #         zip(self.params_gradients, self.network_params)
        #     )
        #     self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

        self.state, self.action_weights, self.actor_model = self.create_actor_model()
        self.network_params = self.actor_model.get_weights()
        #target network
        self.target_state, self.target_action_weights, self.target_actor_model = self.create_actor_model()
        self.target_network_params =self.target_actor_model.get_weights()



        # operator for periodically updating target network with estimator network weights
        self.update_target_network_params = [
            self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1 - self.tau)
            ) for i in range(len(self.target_network_params))
        ]
        self.hard_update_target_network_params = [
            self.target_network_params[i].assign(
                self.network_params[i]
            ) for i in range(len(self.target_network_params))
        ]

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.params_gradients, self.network_params)
        )
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    @staticmethod
    def cli_value(x, v):
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        return tf.where(tf.greater(x, y), x, y)

    def _build_net(self, scope):
        """build the tensorflow graph"""
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            # state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
            # len_seq = tf.placeholder(tf.int32, [None])
            # cell = tf.nn.rnn_cell.GRUCell(self.output_size,
            #                               activation=tf.nn.relu,
            #                               kernel_initializer=tf.initializers.random_normal(),
            #                               bias_initializer=tf.zeros_initializer())
            # outputs, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
            layer1 = tf.layers.Dense(32, activation=tf.nn.relu)(state)
            layer2 = tf.layers.Dense(16, activation=tf.nn.relu)(layer1)

            outputs = tf.layers.Dense(self.a_dim, activation=tf.nn.tanh)(layer2)
        return state, outputs

    def create_actor_model(self):
        state = keras.Input(shape=(self.s_dim,))
        h1 = keras.Dense(16, activation='relu')(state)
        h2 = keras.Dense(32, activation='relu')(h1)
        output = keras.Dense(self.a_dim,
                       activation='tanh')(h2)

        model = keras.Model(input=state, output=output)
        adam = keras.Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state,model.output, model


    # def train(self, state, a_gradient, len_seq):
    #     self.sess.run(self.optimizer, feed_dict={self.state: state, self.a_gradient: a_gradient})
    #
    # def predict(self, state):
    #     return self.sess.run(self.action_weights, feed_dict={self.state: state})
    #
    # def predict_target(self, state):
    #     return self.sess.run(self.target_action_weights, feed_dict={self.target_state: state})

    def train(self, state, a_gradient):
        # self.sess.run(self.optimizer, feed_dict={self.state: state, self.a_gradient: a_gradient})

    def predict(self, state):
        return self.actor_model.predict(state)

    def predict_target(self, state):
        return self.target_actor_model.predict(state)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class Critic(object):
    """value function approximator"""

    def __init__(self, sess, s_dim, a_dim, num_actor_vars, weights_len, gamma, tau, learning_rate, scope="critic"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.num_actor_vars = num_actor_vars
        self.weights_len = weights_len
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # estimator critic network
            self.state, self.action, self.q_value = self._build_net("estimator_critic")
            # self.network_params = tf.trainable_variables()[self.num_actor_vars:]
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="estimator_critic")

            # target critic network
            self.target_state, self.target_action, self.target_q_value = self._build_net(
                "target_critic")
            # self.target_network_params = tf.trainable_variables()[(len(self.network_params) + self.num_actor_vars):]
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_critic")

            # operator for periodically updating target network with estimator network weights
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assgin(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
            self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.q_value))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.a_gradient = tf.gradients(self.q_value, self.action)

    @staticmethod
    def cli_value(x, v):
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        return tf.where(tf.greater(x, y), x, y)

    # def _gather_last_output(self, data, seq_lens):
    #     this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
    #     tmp_end = tf.map_fn(lambda x: self.cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
    #     indices = tf.stack([this_range, tmp_end], axis=1)
    #     return tf.gather_nd(data, indices)


    # def _build_net(self, scope):
    #     with tf.variable_scope(scope):
    #         state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
    #         state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
    #         action = tf.placeholder(tf.float32, [None, self.a_dim], "action")
    #         len_seq = tf.placeholder(tf.int64, [None], name="critic_len_seq")
    #         cell = tf.nn.rnn_cell.GRUCell(self.weights_len,
    #                                       activation=tf.nn.relu,
    #                                       kernel_initializer=tf.initializers.random_normal(),
    #                                       bias_initializer=tf.zeros_initializer()
    #                                       )
    #         out_state, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
    #         out_state = self._gather_last_output(out_state, len_seq)
    #
    #         inputs = tf.concat([out_state, action], axis=-1)
    #         layer1 = tf.layers.Dense(32, activation=tf.nn.relu)(inputs)
    #         layer2 = tf.layers.Dense(16, activation=tf.nn.relu)(layer1)
    #         q_value = tf.layers.Dense(1)(layer2)
    #         return state, action, q_value, len_seq

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            # state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
            action = tf.placeholder(tf.float32, [None, self.a_dim], "action")
            # len_seq = tf.placeholder(tf.int64, [None], name="critic_len_seq")
            # cell = tf.nn.rnn_cell.GRUCell(self.weights_len,
            #                               activation=tf.nn.relu,
            #                               kernel_initializer=tf.initializers.random_normal(),
            #                               bias_initializer=tf.zeros_initializer()
            #                               )
            # out_state, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
            #             # out_state = self._gather_last_output(out_state, len_seq)

            out_state = tf.layers.Dense(self.s_dim, activation=tf.nn.relu)(state)

            inputs = tf.concat([out_state, action], axis=-1)
            layer1 = tf.layers.Dense(32, activation=tf.nn.relu)(inputs)
            layer2 = tf.layers.Dense(16, activation=tf.nn.relu)(layer1)
            q_value = tf.layers.Dense(1)(layer2)
            return state, action, q_value


    def predict(self, state, action, len_seq):
        return self.sess.run(self.q_value, feed_dict={self.state: state, self.action: action})

    def predict_target(self, state, action, len_seq):
        return self.sess.run(self.target_q_value, feed_dict={self.target_state: state, self.target_action: action})

    def action_gradients(self, state, action, len_seq):
        return self.sess.run(self.a_gradient, feed_dict={self.state: state, self.action: action})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)

class RelayBuffer(object):
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()
        self.count = 0
class Noise:
    """generate noise for action"""

    def __init__(self, a_dim, mu=0, theta=0.5, sigma=0.2):
        self.a_dim = a_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.a_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.a_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(len(x))
        self.state = x + dx
        return self.state





