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
from rank_metric import precision_at_k,ndcg_at_k
from popularity_model import PopularityRecommender
from content_model import contentModel
from DDPG_Agent_TF import Agent
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#
# def test_env_agent_offline():
#     path = 'dataset/'
#     features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
#     with tf.Session() as sess:
#         train_mode = True
#         test_mode = False
#         offline_mode = True
#
#         #build enviroment
#
#
#         # this mean the number of items in the recommendation return from the agent
#         slate_size = 5
#
#         # i am assuming this number mean the # of possible items to send to the agent for recommend for each slate
#         num_candidates = 20
#
#         format_data = data_preprocess.load_data(path)
#         features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
#         positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data)
#         # generate train and test set
#         train_set, test_set = data_preprocess.generate_train_test_data(positive_history_data)
#         users_history_data, train_set = data_preprocess.create_recent_history(train_set,
#                                                                               embedding_features_data=features_embedding_movies)
#
#         offline_mode = True
#         rating_pivot = 4
#         sampler = LTSDocumentSampler(dataset=features_embedding_movies,num_candidate=num_candidates)
#         user_sampler = LTSStaticUserSampler(users_history_data, features_embedding_movies, offline_data=test_set,
#                                             offline_mode=offline_mode)
#         # need to handle where we update dataset with num candidate< available
#         func = select_dataset(features_embedding_movies, test_set)
#         LTSUserModel = UserModel(user_sampler, offline_mode=offline_mode, rating_pivot=rating_pivot,
#                                  slate_size=slate_size, response_ctor=LTSResponse)
#         resample_documents = True
#
#
#         ltsenv = CustomSingleUserEnviroment(
#             LTSUserModel,
#             sampler,
#             num_candidates,
#             slate_size,
#             resample_documents=resample_documents, offline_mode=offline_mode, select_subset_func=func)
#         lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)
#         # simulated environment
#
#         #build agent
#         embedding_size = 30
#         num_positive_hisotry_items = 10
#         num_action_vector = 1
#         s_dim = num_positive_hisotry_items*embedding_size
#         a_dim = num_action_vector*embedding_size
#         lr = 0.001
#         tau = 0.2
#         batch_size = 32
#         gamma = 0.125
#         buffer_size = 1000
#         actor = Actor(sess, s_dim, a_dim,batch_size, slate_size,embedding_size, tau,lr)
#         critic = Critic(sess,s_dim,a_dim,slate_size,embedding_size,gamma,tau,lr)
#         buffer = RelayBuffer(buffer_size,s_dim,a_dim)
#         noise_model = Noise(a_dim)
#         agent = Actor_Critic_Agent(sess, lts_gym_env.observation_space, lts_gym_env.action_space, actor, critic, buffer,
#                                    noise_model, slate_size, embedding_size)
#
#         final_p_k = 0
#         final_ndcg_k = 0
#         episode = 50
#         train_episode = 0
#         for eps in range(episode):
#             eps_precision_k = 0.0
#             eps_ndcg_k = 0.0
#             length = 0
#             #start the connection
#             observation_0 = lts_gym_env.reset()
#             done = False
#             # print(observation_0['user'].keys())
#             print("current user : ", observation_0['user']['user_id'])
#             # if (len(observation_0['user']['record_ids'])< 13):
#             #     print("current history of user items :", observation_0['user']['record_ids'])
#             #     print("candidate recommend docs ids : ", observation_0['doc'].keys())
#
#             if (len(observation_0['doc'].keys())<slate_size):
#                 print("can not train with this user ")
#                 done = True
#             reward = 0
#             while (not done):
#                 recommendation_slate_0 = agent.step(reward,observation_0)
#                 print("current recommended slate : ",recommendation_slate_0)
#                 observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)
#                 rate_score_list = [response['rating'] for response in observation_1['response']]
#                 click_list = [int(response['click']) for response in observation_1['response']]
#                 precision_k = precision_at_k(click_list,slate_size)
#                 ndcg_k = ndcg_at_k(rate_score_list,slate_size,method=0)
#                 print("click   : ",click_list)
#                 print("rate score : ",rate_score_list)
#                 print("current precision k :",precision_k)
#                 eps_precision_k+=precision_k
#                 eps_ndcg_k+=ndcg_k
#                 length+=1
#
#                 # print("response : ", observation_1['response'])
#                 # print("next history of recommend items :", observation_1['user']['record_ids'])
#                 # print("total remaind candidate items to recommend : ", len(observation_1['doc'].keys()))
#                 # print("docs ids : ", observation_1['doc'].keys())
#
#                 observation_0 = observation_1
#             if (length > 0):
#                 eps_precision_k = eps_precision_k/length
#                 eps_ndcg_k = eps_ndcg_k/length
#                 train_episode+=1
#                 print("episode p@k :",eps_precision_k)
#                 print("episode ndcg@k : ",eps_ndcg_k)
#
#             final_p_k+= eps_precision_k
#             final_ndcg_k += eps_ndcg_k
#         print("final p@k :", (final_p_k / train_episode))
#         print("final ndcg@k : ", (final_ndcg_k / train_episode))



# def test_env_agent_online():


# test_env_agent_offline()
def generate_config_file():
    return


def test_train_agent_offline():
    path = 'dataset'
    use_100k = True
    if use_100k:
        rating_file_name = 'process_small_rating.csv'
        embedding_file_name = 'movie_embedding_features.csv'
    else:
        rating_file_name = 'process_1m_rating.csv'
        embedding_file_name = 'movie_embedding_features_1m.csv'


    np.random.seed(0)
    # list_slate = [3,5,7,9]
    # list_time_budget = [2,4,6,8]
    # list_num_candidate = [20,30]

    list_slate = [5]
    list_time_budget = [4]
    list_num_candidate = [20]
    slate_size = 5
    time_budget = -4
    time_budget_range = [2,6]
    num_candidates = 30
    min_num_positive_rating = 40
    min_num_rating = 70
    test_mode = False
    offline_mode = True

    rating_pivot = 4
    resample_documents = True

    #agent params
    embedding_size = 30
    num_positive_hisotry_items = 10
    num_action_vector = 1
    s_dim = num_positive_hisotry_items * embedding_size
    a_dim = num_action_vector * embedding_size
    actor_lr = 0.001
    critic_lr = 0.001
    hidden_layer_1 = 32
    hidden_layer_2 = 16
    tau = 0.01
    batch_size = 32
    gamma = 0.75
    buffer_size = 20000

    # train agent
    max_eps = 50
    sample_user_randomly = False
    max_num_user = 100
    save_frequenly = 10
    model_folder = "model_test"
    # config_file =
    save_model_path = "save_model"
    log_path = "logs/scalars/"
    history_path = "history_log"
    model_count = 0
    # for slate_size in list_slate:
    #     for time_budget in list_time_budget:
    #         for num_candidates in list_num_candidate:
    #             curret_model_path = os.path.join(save_model_path, str(slate_size), str(time_budget), str(num_candidates))
    #             current_log_path = os.path.join(log_path, str(slate_size), str(time_budget), str(num_candidates))
    #             current_history_path = os.path.join(history_path, str(slate_size), str(time_budget), str(num_candidates))
    curret_model_path = os.path.join(model_folder,save_model_path)
    current_log_path =  os.path.join(model_folder,log_path)
    current_history_path = os.path.join(model_folder,history_path)
    if not os.path.exists(curret_model_path):
        print("make model path")
        os.makedirs(curret_model_path)
    if not os.path.exists(current_log_path):
        os.makedirs(current_log_path)
    if not os.path.exists(current_history_path):
        os.makedirs(current_history_path)
    with tf.Session() as sess:
        # train_mode = True

        # build enviroment
        # this mean the number of items in the recommendation return from the agent
        # i am assuming this number mean the # of possible items to send to the agent for recommend for each slate
        # time_budget = 2

        format_data = data_preprocess.load_data(path, file_name=rating_file_name)
        features_embedding_movies = pd.read_csv(os.path.join(path, embedding_file_name))
        positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data,min_num_positive_rating)
        print("unique user id : ", len(np.unique(positive_user_ids)))
        print(len(positive_history_data))

        # restrict number of total rating
        positive_user_ids,positive_history_data = data_preprocess.generate_new_dataset(positive_history_data,num_rating_pivot=min_num_rating)
        print("unique user id : ", len(np.unique(positive_user_ids)))
        print(len(positive_history_data))
        # generate train and test set

        train_set, test_set = data_preprocess.generate_train_test_data(positive_history_data)
        users_history_data, train_set = data_preprocess.create_recent_history(train_set)
        #check the train set and test set quality
        user_details = train_set.groupby('userId').size().reset_index()
        user_details.columns = ['userId', 'number of rating']
        print("train set quality : ",user_details.describe())
        user_details = test_set.groupby('userId').size().reset_index()
        user_details.columns = ['userId', 'number of rating']
        print("test set quality : ",user_details.describe())

        offline_dataset = train_set
        if test_mode:
            offline_dataset = test_set


        sampler = LTSDocumentSampler(dataset=features_embedding_movies, num_candidate=num_candidates)
        user_sampler = LTSStaticUserSampler(users_history_data, features_embedding_movies, offline_data=offline_dataset,
                                            offline_mode=offline_mode,time_budget=time_budget,random = sample_user_randomly,time_budget_range=time_budget_range)
        # need to handle where we update dataset with num candidate< available


        func_select_train_set = select_dataset(features_embedding_movies, train_set)
        func_select_test_set = select_dataset(features_embedding_movies, test_set)


        # user_train_set = func(user_id=39)
        # print(len(user_train_set))


        LTSUserModel = UserModel(user_sampler, offline_mode=offline_mode, rating_pivot=rating_pivot,
                                 slate_size=slate_size, response_ctor=LTSResponse)

        ltsenv = CustomSingleUserEnviroment(
            LTSUserModel,
            sampler,
            num_candidates,
            slate_size,
            resample_documents=resample_documents, offline_mode=offline_mode, select_subset_func=func_select_train_set)

        if test_mode:
            ltsenv = CustomSingleUserEnviroment(
                LTSUserModel,
                sampler,
                num_candidates,
                slate_size,
                resample_documents=resample_documents, offline_mode=offline_mode,
                select_subset_func=func_select_test_set)

        lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)
        # simulated environment

        # build agent
        actor = Actor(sess, s_dim, a_dim, batch_size, slate_size, embedding_size, tau, actor_lr,hidden_layer_1,hidden_layer_2)
        critic = Critic(sess, s_dim, a_dim, slate_size, embedding_size, gamma, tau, critic_lr,hidden_layer_1,hidden_layer_2)
        buffer = RelayBuffer(buffer_size,s_dim,a_dim)
        noise_model = Noise(a_dim)
        agent = Actor_Critic_Agent(sess, lts_gym_env.observation_space, lts_gym_env.action_space, actor, critic,
                                   buffer,
                                   noise_model, slate_size, embedding_size)
        #train section
        # max_num_user = len(np.unique(users_history_data['userId']))

        history = agent.train(max_eps,max_num_user,batch_size,lts_gym_env,save_frequenly,curret_model_path,current_log_path,current_history_path)
        #
        # # print(history.keys())
        # history_table = pd.DataFrame(history)
        # history_table.to_csv(os.path.join(current_history_path,"history_record.csv"),index=False)
        #
        # print("finish training for model : ",model_count )
        # model_count += 1
                    #evaluate section
        config_info = {
            "use_teriminal_info":True,
            "use_100k":use_100k,
            "slate_size":slate_size,
            "num_candidates":num_candidates,
            "time_budget":time_budget,
            "time_budget_range":time_budget_range,
            "min_num_rating":min_num_rating,
            "min_num_positive_rating": min_num_positive_rating,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "hidden_layer_1": hidden_layer_1,
            "hidden_layer_2": hidden_layer_2,
            "batch_size": batch_size,
            "tau": tau,
            "gamma":gamma,
            "buffer_size":buffer_size,
            "max_eps":max_eps,
            "max_num_user":max_num_user,
            "sample_user_randomly":sample_user_randomly,
            "save_frequenly":save_frequenly,
        }
        config_file_name = 'config.json'

        with open(os.path.join(model_folder,config_file_name), 'w') as fp:
            json.dump(config_info, fp, indent=4)



# def train_tf_agent_offline():
#     path = 'dataset'
#     features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
#     np.random.seed(0)
#     #
#     # list_slate = [5]
#     # list_time_budget = [4]
#     # list_num_candidate = [20]
#     num_candidates = 20
#     slate_size = 5
#     time_budget = 4
#     offline_mode = True
#     sample_user_randomly = False
#     rating_pivot = 4
#     resample_documents = True
#
#     # agent params
#     embedding_size = 30
#     num_positive_hisotry_items = 10
#     num_action_vector = 1
#     s_dim = num_positive_hisotry_items * embedding_size
#     a_dim = num_action_vector * embedding_size
#     lr = 0.001
#     # critic_lr = 0.0001
#     tau = 0.01
#     batch_size = 32
#     gamma = 0.75
#     buffer_size = 100000
#
#     # train agent
#     max_eps = 50
#     save_frequenly = 5
#     save_model_path = "save_model"
#     log_path = "logs/scalars/"
#     history_path = "history_log"
#
#     format_data = data_preprocess.load_data(path)
#     features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
#     positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data)
#     # generate train and test set
#
#     train_set, test_set = data_preprocess.generate_train_test_data(positive_history_data)
#     users_history_data, train_set = data_preprocess.create_recent_history(train_set,
#                                                                           embedding_features_data=features_embedding_movies)
#     offline_dataset = train_set
#
#     sampler = LTSDocumentSampler(dataset=features_embedding_movies, num_candidate=num_candidates)
#     user_sampler = LTSStaticUserSampler(users_history_data, features_embedding_movies, offline_data=offline_dataset,
#                                         offline_mode=offline_mode, time_budget=time_budget, random=sample_user_randomly)
#     # need to handle where we update dataset with num candidate< available
#
#     func_select_train_set = select_dataset(features_embedding_movies, train_set)
#
#     LTSUserModel = UserModel(user_sampler, offline_mode=offline_mode, rating_pivot=rating_pivot,
#                              slate_size=slate_size, response_ctor=LTSResponse)
#
#     ltsenv = CustomSingleUserEnviroment(
#         LTSUserModel,
#         sampler,
#         num_candidates,
#         slate_size,
#         resample_documents=resample_documents, offline_mode=offline_mode, select_subset_func=func_select_train_set)
#
#     lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)
#
#
#     with tf.Session() as sess:
#         agent = Agent(sess,lr, lr, s_dim,a_dim, slate_size,embedding_size,tau, gamma=gamma,
#                      max_size=buffer_size, layer1_size=400, layer2_size=300,
#                       chkpt_dir='tmp/ddpg')
#         max_num_user = len(np.unique(users_history_data['userId']))
#
#         history = agent.train(max_eps,max_num_user,batch_size,lts_gym_env,save_model_frequenly = 5,save_path = "",log_path = "logs/scalars/",history_path = "history_log")
    # train section



# test_train_agent_offline()

# test_env_agent_offline()


def evaluate_agent_offline(agent,slate_size,max_num_user,lts_gym_env):
    # path = '../master_capston/the-movies-dataset/'
    # features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))

        # simulated environment
        # build agent
        final_p_k = 0
        final_ndcg_k = 0
        train_episode = 0
        for i in range(max_num_user):
            eps_precision_k = 0.0
            eps_ndcg_k = 0.0
            length = 0
            # start the connection
            observation_0 = lts_gym_env.reset()
            done = False
            # print(observation_0['user'].keys())
            # print("current user : ", observation_0['user']['user_id'])
            # if (len(observation_0['user']['record_ids'])< 13):
            #     print("current history of user items :", observation_0['user']['record_ids'])
            #     print("candidate recommend docs ids : ", observation_0['doc'].keys())

            if (len(observation_0['doc'].keys()) < slate_size):
                print("can not train with this user ")
                done = True
            reward = 0
            while (not done):
                recommendation_slate_0 = agent.step(reward, observation_0)
                # print("current recommended slate : ", recommendation_slate_0)
                observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)
                rate_score_list = [response['rating'] for response in observation_1['response']]
                click_list = [int(response['click']) for response in observation_1['response']]
                precision_k = precision_at_k(click_list, slate_size)
                ndcg_k = ndcg_at_k(rate_score_list, slate_size, method=0)
                # print("click   : ", click_list)
                # print("rate score : ", rate_score_list)
                # print("current precision k :", precision_k)
                eps_precision_k += precision_k
                eps_ndcg_k += ndcg_k
                length += 1

                # print("response : ", observation_1['response'])
                # print("next history of recommend items :", observation_1['user']['record_ids'])
                # print("total remaind candidate items to recommend : ", len(observation_1['doc'].keys()))
                # print("docs ids : ", observation_1['doc'].keys())

                observation_0 = observation_1
            if (length > 0):
                eps_precision_k = eps_precision_k / length
                eps_ndcg_k = eps_ndcg_k / length
                train_episode += 1
                print("episode p@k :", eps_precision_k)
                print("episode ndcg@k : ", eps_ndcg_k)

            final_p_k += eps_precision_k
            final_ndcg_k += eps_ndcg_k
        print("final p@k :", (final_p_k / train_episode))
        print("final ndcg@k : ", (final_ndcg_k / train_episode))

def create_RL_Agent(sess,lts_gym_env,slate_size,save_model_path):
    embedding_size = 30
    num_positive_hisotry_items = 10
    num_action_vector = 1
    s_dim = num_positive_hisotry_items * embedding_size
    a_dim = num_action_vector * embedding_size
    lr = 0.001
    tau = 0.001
    batch_size = 32
    gamma = 0.75
    buffer_size = 5000
    actor = Actor(sess, s_dim, a_dim, batch_size, slate_size, embedding_size, tau, lr)
    critic = Critic(sess, s_dim, a_dim, slate_size, embedding_size, gamma, tau, lr)
    buffer = RelayBuffer(buffer_size,s_dim,a_dim)
    noise_model = Noise(a_dim)
    agent = Actor_Critic_Agent(sess, lts_gym_env.observation_space, lts_gym_env.action_space, actor, critic,
                               buffer,
                               noise_model, slate_size, embedding_size)

    agent.load_model(save_model_path)
    return agent

# def create_popularity_recommender(slate_size,dataset):
#     path = '../master_capston/the-movies-dataset/'
#     format_data = data_preprocess.load_data(path)
#     features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
#     positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data)
#     # generate train and test set
#     # train_set, test_set = data_preprocess.generate_train_test_data(positive_history_data)
#     # users_history_data, train_set = data_preprocess.create_recent_history(train_set,
#     #                                                                       embedding_features_data=features_embedding_movies)
#     return PopularityRecommender(dataset, slate_size)

def main():
    path = 'dataset/'
    use_100k = True
    if use_100k:
        rating_file_name = 'process_small_rating.csv'
        embedding_file_name = 'movie_embedding_features.csv'
    else:
        rating_file_name = 'process_1m_rating.csv'
        embedding_file_name = 'movie_embedding_features_1m.csv'

    # train_mode = True
    np.random.seed(0)
    test_mode = True
    offline_mode = True
    sample_user_randomly = False
    save_model_path = "model_test/save_model/49"
    # build enviroment

    # this mean the number of items in the recommendation return from the agent
    slate_size = 5

    # i am assuming this number mean the # of possible items to send to the agent for recommend for each slate
    num_candidates = 30

    rating_pivot = 4
    time_budget = -1
    time_budget_range = [2,6]
    min_num_positive_rating = 40
    min_num_rating = 70
    resample_documents = True

    format_data = data_preprocess.load_data(path,file_name=rating_file_name)
    features_embedding_movies = pd.read_csv(os.path.join(path,embedding_file_name))
    positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data,number_positive_pivot=min_num_positive_rating)
    print("unique user id : ",len(np.unique(positive_user_ids)))
    print(len(positive_history_data))

    # restrict number of total rating
    positive_user_ids, positive_history_data = data_preprocess.generate_new_dataset(positive_history_data,
                                                                                    num_rating_pivot=min_num_rating)
    print("unique user id : ", len(np.unique(positive_user_ids)))
    print(len(positive_history_data))

    # generate train and test set
    # user_details = positive_history_data.groupby('userId').size().reset_index()
    # user_details.columns = ['userId', 'number of rating']
    # print(user_details.describe())

    train_set, test_set = data_preprocess.generate_train_test_data(positive_history_data)
    print("train set size : ",len(train_set))
    print("test set size : ",len(test_set))
    users_history_data, train_set = data_preprocess.create_recent_history(train_set)
    train_set = train_set.astype({'rating': 'float64'})
    test_set = test_set.astype({'rating': 'float64'})
    print("user history set size : ", len(users_history_data))
    print("new train set size : ", len(train_set))
    user_details = test_set.groupby('userId').size().reset_index()
    user_details.columns = ['userId', 'number of rating']
    print(user_details.describe())


    # check the train set and test set quality

    offline_dataset = train_set
    if test_mode:
        offline_dataset = test_set

    sampler = LTSDocumentSampler(dataset=features_embedding_movies, num_candidate=num_candidates)
    user_sampler = LTSStaticUserSampler(users_history_data, features_embedding_movies, offline_data=offline_dataset,
                                        offline_mode=offline_mode, time_budget=time_budget, random=sample_user_randomly,time_budget_range=time_budget_range)
    # need to handle where we update dataset with num candidate< available

    func_select_train_set = select_dataset(features_embedding_movies, train_set)
    func_select_test_set = select_dataset(features_embedding_movies, test_set)

    # user_train_set = func(user_id=39)
    # print(len(user_train_set))

    LTSUserModel = UserModel(user_sampler, offline_mode=offline_mode, rating_pivot=rating_pivot,
                             slate_size=slate_size, response_ctor=LTSResponse)

    ltsenv = CustomSingleUserEnviroment(
        LTSUserModel,
        sampler,
        num_candidates,
        slate_size,
        resample_documents=resample_documents, offline_mode=offline_mode, select_subset_func=func_select_train_set)

    if test_mode:
        ltsenv = CustomSingleUserEnviroment(
            LTSUserModel,
            sampler,
            num_candidates,
            slate_size,
            resample_documents=resample_documents, offline_mode=offline_mode,
            select_subset_func=func_select_test_set)

    lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)
    max_num_user = len(np.unique(users_history_data['userId']))

    with tf.Session() as sess:
        popularAgent= PopularityRecommender(train_set,slate_size,"1")

        # RL_Agent = create_RL_Agent(sess,lts_gym_env,slate_size,save_model_path=save_model_path)
        # contentAgent = contentModel(slate_size,embedding_size=30)


        evaluate_agent_offline(popularAgent,slate_size,max_num_user,lts_gym_env)
        # evaluate_agent_offline(RL_Agent,slate_size,max_num_user,lts_gym_env)
        # evaluate_agent_offline(contentAgent,slate_size,max_num_user,lts_gym_env)


main()
# test_train_agent_offline()

# train_tf_agent_offline()

