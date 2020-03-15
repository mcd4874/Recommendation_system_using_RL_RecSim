

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

    def __init__(self, sess, s_dim, a_dim, batch_size, output_size, weights_len, tau, learning_rate):
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

        self.actor_state_input,self.actor_model = self.create_actor_model()
        _,self.target_actor_model = self.create_actor_model()


        self.optimizer = self.create_optimizer()
        # print('actor weights ',self.actor_model.get_weights())
        self.num_trainable_vars = len(self.actor_model.trainable_weights) + len(self.target_actor_model.trainable_weights)

    def create_actor_model(self):
        state = keras.Input(shape=(self.s_dim,))
        h1 = keras.layers.Dense(16, activation='relu')(state)
        h2 = keras.layers.Dense(32, activation='relu')(h1)
        output = keras.layers.Dense(self.a_dim,
                       activation='tanh')(h2)

        model = keras.Model(state,output)
        return state,model
    def create_optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.a_dim))
        params_grad = tf.gradients(self.actor_model.output, self.actor_model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor_model.trainable_weights)
        return K.function([self.actor_model.input, action_gdts], [tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)])

        # self.actor_critic_grad = tf.placeholder(tf.float32,
        #                                         [None, self.a_dim])  # where we will feed de/dC (from critic)
        #
        # actor_model_weights = self.actor_model.trainable_weights
        #
        # print("actor model weights : ",actor_model_weights)
        # self.actor_grads = tf.gradients(self.actor_model.output,
        #                                 actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        # grads = zip(self.actor_grads, actor_model_weights)
        # return tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    def train(self, states,grads):
        """ Actor Training
                """
        # self.optimizer([states, grads])


        self.sess.run(self.optimizer, feed_dict={
            self.actor_state_input: states,
            self.actor_critic_grad: grads
        })

    def predict(self, state):
        return self.actor_model.predict(state)

    def predict_target(self, state):
        return self.target_actor_model.predict(state)

    def update_target_network(self):

        """ Transfer model weights to target model with a factor of Tau
                """
        W, target_W = self.actor_model.get_weights(), self.target_actor_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_actor_model.set_weights(target_W)


    def hard_update_target_network(self):
        W, target_W = self.actor_model.get_weights(), self.target_actor_model.get_weights()
        for i in range(len(W)):
            target_W[i] = W[i]
        self.target_actor_model.set_weights(target_W)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def save(self, path):
        self.actor_model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.actor_model.load_weights(path)


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
        self.critic_model = self.create_critic_model()
        self.target_critic_model = self.create_critic_model()
        self.critic_model.compile(keras.optimizers.Adam(lr=self.learning_rate), 'mse')
        self.target_critic_model.compile(keras.optimizers.Adam(lr=self.learning_rate), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.critic_model.input[0], self.critic_model.input[1]],
                                       K.gradients(self.critic_model.output, [self.critic_model.input[1]]))


    def create_critic_model(self):
        state = keras.Input(shape=(self.s_dim,))
        action = keras.Input(shape=(self.a_dim,))
        state_transform = keras.layers.Dense(self.s_dim, activation = 'relu')(state)
        input_state = keras.layers.concatenate([state_transform,action])
        h1 = keras.layers.Dense(16, activation='relu')(input_state)
        h2 = keras.layers.Dense(32, activation='relu')(h1)
        output = keras.layers.Dense(1)(h2)

        model = keras.Model([state,action],output)
        return model

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])
    def train(self,states,actions,target_critic_values):
        return self.critic_model.train_on_batch([states, actions], target_critic_values)

    def predict(self, state, action):
        return self.critic_model.predict([state, action])

    def predict_target(self, states, actions):
        """ Predict Q-Values using the target network
                """
        return self.target_critic_model.predict([states,actions])

    def update_target_network(self):

        """ Transfer model weights to target model with a factor of Tau
                """
        W, target_W = self.critic_model.get_weights(), self.target_critic_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_critic_model.set_weights(target_W)


    def hard_update_target_network(self):
        W, target_W = self.critic_model.get_weights(), self.target_critic_model.get_weights()
        for i in range(len(W)):
            target_W[i] = W[i]
        self.target_critic_model.set_weights(target_W)

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





