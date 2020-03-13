import functools
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# RecSim imports
from recsim import agent
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym
from recsim.simulator import runner_lib
from recsim.agent import AbstractEpisodicRecommenderAgent
from model import Actor,Critic, Noise,RelayBuffer
from enviroment import UserModel,LTSDocumentSampler,LTSStaticUserSampler,LTSResponse,clicked_engagement_reward
import pandas as pd
import os
import data_preprocess
# Just disables the warning, doesn't enable AVX/FMA
from model import Actor,Critic
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Actor_Critic_Agent(AbstractEpisodicRecommenderAgent):
    def __init__(self, sess,
               observation_space,
               action_space,actor_model, critic_model,buffer,noise_model,
               optimizer_name='',
               eval_mode=False,
               **kwargs):
    # Check if document corpus is large enough.
        if len(observation_space['doc'].spaces) < len(action_space.nvec):
            print("problem happens")
            raise RuntimeError('Slate size larger than size of the corpus.')
        super(Actor_Critic_Agent, self).__init__(action_space)

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.buffer = buffer
        self.noise_model = noise_model
        self.sess = sess
        # self.buffer_size = 300
        print('observe space ',observation_space)
        print('action space ',action_space)


    def generate_doc_representation(self,doc):
        origin_corpus = list(doc.values())
        # print(origin_corpus[0].shape)
        corpus = np.array(origin_corpus).reshape((len(origin_corpus), origin_corpus[0].shape[1]))
        corpus_key = np.array(list(doc.keys()))
        return [corpus,corpus_key]

    def generate_state_represetation(self,user):
        user_state = user.reshape(1,user.shape[0]*user.shape[1])
        return user_state


    def step(self, reward, observation):
        print("stat to step ")

        doc = observation['doc']
        user = observation['user']
        items_space,item_keys = self.generate_doc_representation(doc)
        state = self.generate_state_represetation(user)
        print('user infor: ',state.shape)
        print('items space info :',items_space.shape)
        print('items keys info :',item_keys.shape)
        # print(states.shape)
        # print(doc)
        # print(doc[0])

        actions = self.actor_model.predict(state)
        result_recommend_list = self.generate_actions(items_space,actions)
        return item_keys[result_recommend_list]

    def update_network_from_batches(self,batch_size):
        if self.buffer.size() < batch_size:
            pass
        samples = self.buffer.sample_batch(batch_size)
        state_batch = np.asarray([_[0] for _ in samples])
        action_batch = np.asarray([_[1] for _ in samples])
        reward_batch = np.asarray([_[2] for _ in samples])
        n_state_batch = np.asarray([_[3] for _ in samples])

        # calculate predicted q value
        action_weights = self.actor_model.predict_target(state_batch)
        # n_action_batch = gene_actions(item_space, action_weights, action_len)
        target_q_batch = self.critic_model.predict_target(n_state_batch, action_weights)
        y_batch = []
        for i in range(batch_size):
            y_batch.append(reward_batch[i] + self.critic_model.gamma * target_q_batch[i])

        # train critic
        q_value, critic_loss, _ = self.critic_model.train(state_batch, action_batch, np.reshape(y_batch, (batch_size, 1)))
        # train actor
        action_weight_batch_for_gradients = self.actor_model.predict(state_batch)
        # action_batch_for_gradients = gene_actions(item_space, action_weight_batch_for_gradients, action_len)
        a_gradient_batch = self.critic_model.action_gradients(state_batch,action_weight_batch_for_gradients)
        self.actor_model.train(state_batch, a_gradient_batch[0])

        # update target networks
        self.actor_model.update_target_network()
        self.critic_model.update_target_network()

        return np.amax(q_value), critic_loss

    def train(self,max_episode,episode_length,batch_size,env):
        # initialize target network weights
        self.actor_model.hard_update_target_network()
        self.critic_model.hard_update_target_network()

        for eps in range(max_episode):
            ep_reward = 0.
            ep_q_value = 0.
            loss = 0.
            # item_space = recall_data
            observation = env.reset()
            doc = observation['doc']
            user = observation['user']
            items_space, item_keys = self.generate_doc_representation(doc)
            state = self.generate_state_represetation(user)
            for step in range(episode_length):
                actions = self.actor_model.predict(state)+self.noise_model.noise()
                result_recommend_list = self.generate_actions(items_space, actions)
                reward, next_observation = env.step(result_recommend_list)
                next_state = self.generate_state_represetation(next_observation['user'])
                self.buffer.add(state,actions,reward,next_state)
                ep_reward += reward

                ep_q_value_, critic_loss = self.update_network_from_batches(batch_size)
                ep_q_value += ep_q_value_
                loss += critic_loss
                state = next_state

                # if (j + 1) % 50 == 0:
                #     logger.info("=========={0} episode of {1} round: {2} reward=========".format(i, j, ep_reward))


        return

    def generate_actions(self,item_space, weight_batch):
        """use output of actor network to calculate action list
        Args:
            item_space: (m,n) where m is # of items and n is embedding size
            weight_batch: actor network outputs (k,n) where k is # of items to pick and n is embedding size
        Returns:
            recommendation list of index (k)
        """
        recommend_items = np.dot(weight_batch,np.transpose(item_space))
        return np.argmax(recommend_items, axis=1)

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("reward", episode_reward)
        episode_max_q = tf.Variable(0.)
        tf.summary.scalar("max_q_value", episode_max_q)
        critic_loss = tf.Variable(0.)
        tf.summary.scalar("critic_loss", critic_loss)
        summary_vars = [episode_reward, episode_max_q, critic_loss]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars
    # def build_network(self):
    #
    # def build_replay_buffer(self):
    #     self.buffer = RelayBuffer(self.buffer_size)
    #
    # def begin_episode(self):
    #
    # def end_episode(self):
    #     hh
    #
    # def learn_from_batch(self,replay_buffer, batch_size,item_space, action_len, s_dim, a_dim):
    #     if replay_buffer.size() < batch_size:
    #         pass
    #     samples = replay_buffer.sample_batch(batch_size)
    #     state_batch = np.asarray([_[0] for _ in samples])
    #     action_batch = np.asarray([_[1] for _ in samples])
    #     reward_batch = np.asarray([_[2] for _ in samples])
    #     n_state_batch = np.asarray([_[3] for _ in samples])
    #
    #     # calculate predicted q value
    #     action_weights = self.actor.predict_target(state_batch)
    #     n_action_batch = gene_actions(item_space, action_weights, action_len)
    #     target_q_batch = self.critic.predict_target(n_state_batch.reshape((-1, s_dim)), n_action_batch.reshape((-1, a_dim)))
    #     y_batch = []
    #     for i in range(batch_size):
    #         y_batch.append(reward_batch[i] + self.critic.gamma * target_q_batch[i])
    #
    #     # train critic
    #     q_value, critic_loss, _ = self.critic.train(state_batch, action_batch, np.reshape(y_batch, (batch_size, 1)))
    #     # train actor
    #     action_weight_batch_for_gradients = self.actor.predict(state_batch)
    #     action_batch_for_gradients = gene_actions(item_space, action_weight_batch_for_gradients, action_len)
    #     a_gradient_batch = self.critic.action_gradients(state_batch, action_batch_for_gradients.reshape((-1, a_dim)))
    #     self.actor.train(state_batch, a_gradient_batch[0])
    #
    #     # update target networks
    #     self.actor.update_target_network()
    #     self.critic.update_target_network()
    #
    #     return np.amax(q_value), critic_loss








path = '../master_capston/the-movies-dataset/'
features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
sampler = LTSDocumentSampler(dataset=features_embedding_movies)
slate_size = 3
num_candidates = 15

format_data = data_preprocess.load_data(path)
# print(format_data.head())
# print(format_data.shape)

features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data)
user_sampler = LTSStaticUserSampler(positive_user_ids, positive_history_data, features_embedding_movies)
LTSUserModel = UserModel(user_sampler, slate_size, LTSResponse)
ltsenv = environment.Environment(
        LTSUserModel,
        sampler,
        num_candidates,
        slate_size,
        resample_documents=True)
lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)


def create_agent(sess, environment, eval_mode, summary_writer=None):
  return Actor_Critic_Agent(environment.observation_space, environment.action_space)

tmp_base_dir = 'detmp/recsim/'

# runner = runner_lib.EvalRunner(
#   base_dir=tmp_base_dir,
#   create_agent_fn=create_agent,
#   env=lts_gym_env,
#   max_eval_episodes=1,
#   max_steps_per_episode=3,
#   test_mode=True)
#
# runner.run_experiment()

def test_agent():
    with tf.Session() as sess:
        # simulated environment
        embedding_size = 30
        num_positive_hisotry_items = 10
        s_dim = num_positive_hisotry_items*embedding_size
        a_dim = slate_size*embedding_size
        lr = 0.001
        tau = 0.2
        batch_size = 32
        gamma = 0.125
        buffer_size = 1000
        actor = Actor(sess, s_dim, a_dim,batch_size, slate_size,embedding_size, tau,lr)
        critic = Critic(sess,s_dim,a_dim,slate_size,embedding_size,gamma,tau,lr)
        buffer = RelayBuffer(buffer_size)
        noise_model = Noise(a_dim)
        agent = Actor_Critic_Agent(sess,lts_gym_env.observation_space,lts_gym_env.action_space,actor,critic,buffer,noise_model)
        observation_0 = lts_gym_env.reset()
        print(observation_0['user'][:5])
        # print('Observation 0')
        # print('Available documents')
        # doc_strings = ['doc_id ' + key + " kaleness " + str(value) for key, value
        #                in observation_0['doc'].items()]
        # print('\n'.join(doc_strings))
        # slates = agent.step(0,observation_0)
        recommendation_slate_0 = [0, 1, 2]
        recommend_slate = agent.step(0,observation_0)
        observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)

test_agent()
