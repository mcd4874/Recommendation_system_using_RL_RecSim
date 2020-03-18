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
from enviroment import UserModel,LTSDocumentSampler,LTSStaticUserSampler,LTSResponse,clicked_engagement_reward,CustomSingleUserEnviroment,select_dataset
import pandas as pd
import os
import data_preprocess
# Just disables the warning, doesn't enable AVX/FMA
from model import Actor,Critic
import os
import tensorflow as tf
from agent import Actor_Critic_Agent
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_env_agent_offline():
    path = '../master_capston/the-movies-dataset/'
    features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
    with tf.Session() as sess:

        sampler = LTSDocumentSampler(dataset=features_embedding_movies)

        # this mean the number of items in the recommendation return from the agent
        slate_size = 3

        # i am assuming this number mean the # of possible items to send to the agent for recommend for each slate
        num_candidates = 11

        format_data = data_preprocess.load_data(path)
        features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
        positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data)
        user_sampler = LTSStaticUserSampler(positive_user_ids, positive_history_data, features_embedding_movies)
        func = select_dataset(features_embedding_movies, positive_history_data)


        LTSUserModel = UserModel(user_sampler, slate_size, LTSResponse)
        ltsenv = CustomSingleUserEnviroment(
            LTSUserModel,
            sampler,
            num_candidates,
            slate_size,
            resample_documents=False, offline_mode=True, select_subset_func=func)
        lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)




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
        agent = Actor_Critic_Agent(sess, lts_gym_env.observation_space, lts_gym_env.action_space, actor, critic, buffer,
                                   noise_model, slate_size, embedding_size)

        observation_0 = lts_gym_env.reset()
        print(observation_0['user'].keys())
        print("current user : ", observation_0['user']['user_id'])
        print("current history of user items :", observation_0['user']['record_ids'])
        print("candidate recommend docs ids : ", observation_0['doc'].keys())
        done = False
        while (not done):
            recommendation_slate_0 = agent.step(0,observation_0)
            observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)
            print("response : ", observation_1['response'])
            print("next history of recommend items :", observation_1['user']['record_ids'])
            print("total remaind candidate items to recommend : ", len(observation_1['doc'].keys()))
            print("docs ids : ", observation_1['doc'].keys())

            observation_0 = observation_1
# def test_env_agent_online():


test_env_agent_offline()