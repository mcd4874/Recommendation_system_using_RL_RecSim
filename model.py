

import os
import argparse
import numpy as np
import tensorflow as tf
import pprint as pp

from collections import deque
import random
from tensorflow import keras as keras
from tensorflow.initializers import random_uniform
# from tensorflow.keras import backend as K
import keras.backend as K
# hidden_1 = 400
# hidden_2 = 300
class Actor(object):
    """policy function approximator"""

    def __init__(self, sess, s_dim, a_dim, batch_size, output_size, weights_len, tau, learning_rate,hidden_layer_1 = 256,hidden_layer_2 = 128):
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
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.batch_size = batch_size
        self.output_size = output_size
        self.weights_len = weights_len
        self.tau = tau
        self.learning_rate = learning_rate

        self.actor_model = self.create_actor_model()
        self.target_actor_model = self.create_actor_model()


        self.optimizer = self.create_optimizer()
        # print('actor weights ',self.actor_model.get_weights())
        self.num_trainable_vars = len(self.actor_model.trainable_weights) + len(self.target_actor_model.trainable_weights)

    def create_actor_model(self):
        state = keras.Input(shape=(self.s_dim,))
        h1 = keras.layers.Dense(self.hidden_layer_1, activation='relu')(state)
        h2 = keras.layers.Dense(self.hidden_layer_2, activation='relu')(h1)
        output = keras.layers.Dense(self.a_dim,
                       activation='tanh')(h2)

        model = keras.Model(state,output)
        model.summary()
        return model

    # def create_actor_model(self):
    #     state = keras.Input(shape=(self.s_dim,))
    #     f1 = 1. / np.sqrt(hidden_1)
    #     h1 = keras.layers.Dense(hidden_1, activation='relu',kernel_initializer=random_uniform(-f1, f1),
    #                                  bias_initializer=random_uniform(-f1, f1))(state)
    #     h1 = keras.layers.BatchNormalization()(h1)
    #     f2 = 1. / np.sqrt(hidden_2)
    #     h2 = keras.layers.Dense(hidden_2, activation='relu',kernel_initializer=random_uniform(-f2, f2),
    #                                  bias_initializer=random_uniform(-f2, f2))(h1)
    #     h2 = keras.layers.BatchNormalization()(h2)
    #     f3 = 0.003
    #     output = keras.layers.Dense(self.a_dim,
    #                    activation='tanh',kernel_initializer= random_uniform(-f3, f3),
    #                         bias_initializer=random_uniform(-f3, f3))(h2)
    #
    #     model = keras.Model(state,output)
    #     model.summary()
    #     return state,model
    def create_optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.a_dim))
        params_grad = tf.gradients(self.actor_model.output, self.actor_model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor_model.trainable_weights)
        return K.function(inputs=[self.actor_model.input, action_gdts], outputs=[],updates=[tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)])
        # return K.function([self.actor_model.input, action_gdts], [tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)])



    def train(self, states,grads):
        """ Actor Training
                """
        # print("a gradient batch : ",np.array(grads).reshape((-1, self.a_dim)))
        grads = np.array(grads).reshape((-1, self.a_dim))
        # print("state train shapes : ",states.shape)
        # print("shape of grad : ",grads.shape)
        self.optimizer([states, grads])

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

    def save(self,path=""):
        self.actor_model.save(os.path.join(path,"actor_model.h5"))

    def load_model(self,path=""):
        self.actor_model = keras.models.load_model(os.path.join(path,"actor_model.h5"))



class Critic(object):
    """value function approximator"""

    def __init__(self, sess, s_dim, a_dim, num_actor_vars, weights_len, gamma, tau, learning_rate,hidden_layer_1 = 256,hidden_layer_2 = 128):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        # self.num_actor_vars = num_actor_vars
        # self.weights_len = weights_len
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
        h1 = keras.layers.Dense(self.hidden_layer_1, activation='relu')(input_state)
        h2 = keras.layers.Dense(self.hidden_layer_2, activation='relu')(h1)
        output = keras.layers.Dense(1,activation='linear', kernel_initializer=random_uniform())(h2)
        model = keras.Model([state,action],output)
        model.summary()
        return model

    # def create_critic_model(self):
    #     state = keras.Input(shape=(self.s_dim,))
    #     action = keras.Input(shape=(self.a_dim,))
    #     state_transform = keras.layers.Dense(self.s_dim, activation = 'relu')(state)
    #     input_state = keras.layers.concatenate([state_transform,action])
    #     f1 = 1. / np.sqrt(hidden_1)
    #     h1 = keras.layers.Dense(units = hidden_1, activation='relu',kernel_initializer=random_uniform(-f1, f1),
    #                                  bias_initializer=random_uniform(-f1, f1))(input_state)
    #     h1 = keras.layers.BatchNormalization()(h1)
    #     f2 = 1. / np.sqrt(hidden_2)
    #     h2 = keras.layers.Dense(units = hidden_2, activation='relu',kernel_initializer=random_uniform(-f2, f2),
    #                                  bias_initializer=random_uniform(-f2, f2))(h1)
    #     h2 = keras.layers.BatchNormalization()(h2)
    #     f3 = 0.003
    #     output = keras.layers.Dense(units = 1,kernel_initializer=random_uniform(-f3, f3),
    #                            bias_initializer=random_uniform(-f3, f3),
    #                            kernel_regularizer=keras.regularizers.l2(0.01))(h2)
    #
    #     model = keras.Model([state,action],output)
    #     model.summary()
    #     return model

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

    def save(self,path=""):
        self.critic_model.save(os.path.join(path,"critic_model.h5"))

# class RelayBuffer(object):
#     def __init__(self, buffer_size=10000):
#         self.buffer_size = buffer_size
#         self.count = 0
#         self.buffer = deque()
#
#     def add(self, state, action, reward, next_state):
#         experience = (state, action, reward, next_state)
#         if self.count < self.buffer_size:
#             self.buffer.append(experience)
#             self.count += 1
#         else:
#             self.buffer.popleft()
#             self.buffer.append(experience)
#
#     def size(self):
#         return self.count
#
#     def sample_batch(self, batch_size):
#         return random.sample(self.buffer, batch_size)
#
#     def clear(self):
#         self.buffer.clear()
#         self.count = 0

class RelayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal = np.zeros(self.mem_size)
        self.current_fill = 0

    def add(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal[index] = 1-done
        self.mem_cntr += 1
        self.current_fill = min((self.current_fill+1),self.mem_size)
    def size(self):
        return self.current_fill


    def sample_batch(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal[batch]

        return states, actions, rewards, states_,terminal
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





