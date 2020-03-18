"""
William Duong

"""

import numpy as np
import pandas as pd
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator.environment import SingleUserEnvironment
from recsim.simulator import recsim_gym
import os
import collections


import data_preprocess

class CustomSingleUserEnviroment(SingleUserEnvironment):
    """Class to represent the custome environment with one user.

      Attributes:
        user_model: An instantiation of AbstractUserModel that represents a user.
        document_sampler: An instantiation of AbstractDocumentSampler.
        num_candidates: An integer representing the size of the candidate_set.
        slate_size: An integer representing the slate size.
        candidate_set: An instantiation of CandidateSet.
        num_clusters: An integer representing the number of document clusters.
      """
    def __init__(self,user_model,
               document_sampler,
               num_candidates,
               slate_size,
               resample_documents=True,
                 offline_mode = False,select_subset_func = None):
        """

        :param user_model:
        :param document_sampler:
        :param num_candidates:
        :param slate_size:
        :param resample_documents:
        :param offline_mode:
        :param select_subset_func:
        """
        super(CustomSingleUserEnviroment, self).__init__(user_model,
               document_sampler,
               num_candidates,
               slate_size,
               resample_documents)
        self.offline_mode = offline_mode
        self.select_subset_func = select_subset_func


    def reset(self):
        """Resets the environment and return the first observation.

        Returns:
          user_obs: An array of floats representing observations of the user's
            current state
          doc_obs: An OrderedDict of document observations keyed by document ids
        """
        self._user_model.reset()
        user_obs = self._user_model.create_observation()

        # only use a user's history record as recommendable items
        if self.offline_mode and self.select_subset_func:
            user_history_records = self.select_subset_func(user_obs['user_id'])
            self._document_sampler.update_dataset(user_history_records)
            print("recommendable doc size: ",self._document_sampler.dataset.shape[0])
        # if self._resample_documents:
        self._do_resample_documents()
        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation())
        return (user_obs, self._current_documents)


    def step(self, slate):
        """Executes the action, returns next state observation and reward.

        Args:
          slate: An integer array of size slate_size, where each element is an index
            into the set of current_documents presented

        Returns:
          user_obs: A gym observation representing the user's next state
          doc_obs: A list of observations of the documents
          responses: A list of AbstractResponse objects for each item in the slate
          done: A boolean indicating whether the episode has terminated
        """

        assert (len(slate) <= self._slate_size
                ), 'Received unexpectedly large slate size: expecting %s, got %s' % (
            self._slate_size, len(slate))

        # Get the documents associated with the slate
        doc_ids = list(self._current_documents)  # pytype: disable=attribute-error
        mapped_slate = [doc_ids[x] for x in slate]
        documents = self._candidate_set.get_documents(mapped_slate)
        # Simulate the user's response
        responses = self._user_model.simulate_response(documents)

        # Update the user's state.
        self._user_model.update_state(documents, responses)

        # Update the documents' state.
        self._document_sampler.update_state(documents, responses)


        # Obtain next user state observation.
        user_obs = self._user_model.create_observation()

        # Check if reaches a terminal state and return.
        done = self._user_model.is_terminal()

        # Optionally, recreate the candidate set to simulate candidate
        # generators for the next query.
        if self._resample_documents:
            self._do_resample_documents()

        #update candidate set based on the response
        self.update_candidate_set(documents, responses)

        # Create observation of candidate set.
        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation())

        return (user_obs, self._current_documents, responses, done)

    def update_candidate_set(self,documens,responses):
        for doc, response in zip(documens, responses):
            if response.clicked:
                self._candidate_set.remove_document(doc)

class LTSDocument(document.AbstractDocument):
    def __init__(self, doc_id, embedding_features):
        # print("input doc_id ",doc_id)
        self.embedding_features = embedding_features
        # doc_id is an integer representing the unique ID of this document
        super(LTSDocument, self).__init__(doc_id)

    def create_observation(self):
        return self.embedding_features


    def observation_space(self):
        return spaces.Box(shape=(len(self.embedding_features),), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        return "Document {} ".format(self._doc_id)
    def get_doc_id(self):
        return self._doc_id


class LTSDocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, dataset ,doc_ctor=LTSDocument, **kwargs):
        super(LTSDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self.dataset = dataset
        self.count = 0

    def sample_document(self):
        columns = self.dataset.columns
        current_item = self.dataset[self.count:self.count+1]
        doc_features = {}
        doc_features['doc_id'] = current_item[columns[0]].values[0]
        doc_features['embedding_features'] = current_item[columns[1:]].values

        self.count = (self.count+1)%self.dataset.shape[0]
        return self._doc_ctor(**doc_features)

    def update_state(self, documents, responses):
        """Update document state (if needed) given user's (or users') responses."""

    def update_dataset(self,dataset):
        self.dataset = dataset

class LTSUserState(user.AbstractUserState):
    def __init__(self, memory_discount,time_budget,user_info,corpus_features_dim = 30,history_record_size = 10):
        ## Transition model parameters
        ##############################
        # self.memory_discount = memory_discount
        # self.sensitivity = sensitivity
        # self.innovation_stddev = innovation_stddev


        ## State variables
        self.time_budget = time_budget
        self.satisfaction = 0
        self.corpus_features_dim = corpus_features_dim
        self.history_record_size = history_record_size

        self.user_id = user_info['userId']
        self.user_recent_past_record_ids = user_info['record_ids']
        self.user_recent_past_record = user_info['past_record']

        # update the user recent history with new recommendation from left to right
        self.state_update_index = 0


    def create_observation(self):
        """User's state is not observable."""
        return {'user_id':self.user_id,'record_ids': np.array(self.user_recent_past_record_ids), 'past_record': np.array(self.user_recent_past_record)}

    def observation_space(self):
        return spaces.Dict({
            'user_id':spaces.Discrete(100000),
            'record_ids':
                spaces.Box(shape=(self.history_record_size,), dtype=np.int8, low=0, high=1000000),
            'past_record':
                spaces.Box(shape=(self.history_record_size, self.corpus_features_dim), dtype=np.float32, low=-10.0, high=10.0)
        })

    # scoring function for use in the choice model -- the user is more likely to
    def score_document(self, doc_obs):

        #jsut sum all the features and avergae for now
        likely = np.sum(doc_obs)
        return 1 - likely

    def update_time_buget(self):
        self.time_budget -=1

    def update_user_history_record(self,new_doc_id, new_doc_feature):
        n = len(self.user_recent_past_record_ids)
        for index in range(1,n):
            self.user_recent_past_record_ids[index-1] = self.user_recent_past_record_ids[index]
            self.user_recent_past_record[index-1] = self.user_recent_past_record[index]
        self.user_recent_past_record_ids[n-1] = new_doc_id
        self.user_recent_past_record[n - 1] = new_doc_feature


class LTSStaticUserSampler(user.AbstractUserSampler):
    def __init__(self, posible_user_ids,user_data,corpus_data,memory_discount=0.9,
               time_budget=4,history_size = 10,doc_feature_size = 30,
               user_ctor=LTSUserState,seed = 0,
               **kwargs):
        super(LTSStaticUserSampler, self).__init__(user_ctor, **kwargs)
        self._state_parameters = {'memory_discount': memory_discount,
                                  'time_budget': time_budget
                                 }
        self.corpus_data = corpus_data
        self.user_data = user_data
        self.posible_user_ids = posible_user_ids
        self.seed = seed
        self.doc_feature_size = doc_feature_size
        self.history_size = history_size

    def sample_user(self):


        # pick one user out of list of possible user to perform the study
        pick_user_id = np.random.choice(self.posible_user_ids, 1)[0]
        history_data = self.user_data[self.user_data['userId'] == pick_user_id].sort_values(by=['timestamp'])
        while(history_data.shape[0] < self.history_size ):
            print('need to resample')
            pick_user_id = np.random.choice(self.posible_user_ids, 1)[0]
            print('user id resample',pick_user_id)
            history_data = self.user_data[self.user_data['userId'] == pick_user_id].sort_values(by=['timestamp'])


        # create a matrix for user history doc features
        past_record = np.zeros((self.history_size,self.doc_feature_size))
        past_record_ids = history_data['movieId'].values[:self.history_size]
        for index in range(self.history_size):
            current_move_id = past_record_ids[index]
            current_movie_embedding_features = self.corpus_data[self.corpus_data['id'] == current_move_id]
            past_record[index] = current_movie_embedding_features.values[0,1:]


        user_info = {
            'userId': pick_user_id,
            'record_ids': past_record_ids,
            'past_record': past_record
        }
        self._state_parameters['user_info'] = user_info
        return self._user_ctor(**self._state_parameters)


class LTSResponse(user.AbstractResponse):
    # The maximum degree of engagement.
    MAX_ENGAGEMENT_MAGNITUDE = 100.0

    def __init__(self, clicked=False, engagement=0.0):
        self.clicked = clicked
        self.engagement = engagement

    def create_observation(self):
        return {'click': int(self.clicked), 'engagement': np.array(self.engagement)}

    @classmethod
    def response_space(cls):
    # `engagement` feature range is [0, MAX_ENGAGEMENT_MAGNITUDE]
        return spaces.Dict({
            'click':
                spaces.Discrete(2),
            'engagement':
                spaces.Box(
                    low=0.0,
                    high=cls.MAX_ENGAGEMENT_MAGNITUDE,
                    shape=tuple(),
                    dtype=np.float32)
        })

class UserModel(user.AbstractUserModel):
    def __init__(self, sampler, slate_size = 10,response_ctor = LTSResponse):
        super(UserModel, self).__init__(response_ctor,sampler,slate_size)
        self.choice_model = MultinomialLogitChoiceModel({})

    def simulate_response(self, slate_documents):
        # List of empty responses
        responses = [self._response_model_ctor() for _ in slate_documents]
        # Get click from of choice model.
        self.choice_model.score_documents(
            self._user_state, [doc.create_observation() for doc in slate_documents])
        scores = self.choice_model.scores
        selected_index = self.choice_model.choose_item()
        # Populate clicked item.
        # print("the select corpus index is :",selected_index)
        # print("length of slate doc :",len(slate_documents))
        self.generate_response(slate_documents[selected_index],
                                responses[selected_index])
        return responses

    def update_state(self, slate_documents, responses):
        """

        :param slate_documents: doc object
        :param responses:  response object
        :return:
        """
        # print("current slate documents to update : ",slate_documents[0])
        # print("current responses to update : ",responses[0].create_observation())
        for doc, response in zip(slate_documents, responses):
            if response.clicked:
                self._user_state.satisfaction = 0
                self._user_state.update_time_buget()
                self._user_state.update_user_history_record(doc.get_doc_id(),doc.create_observation())


    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return self._user_state.time_budget <= 0

    def generate_response(self, doc, response):
        response.clicked = True
        response.engagement = 0



def clicked_engagement_reward(responses):
    """
    this function will calculate the reward
    :param responses:
    :return:
    """
    reward = 0.0
    for response in responses:
        if response.clicked:
          reward += response.engagement
    return reward

def select_dataset(dataset,user_data):
    def user_history_documents(user_id):
        history_data = user_data[user_data['userId'] == user_id].sort_values(by=['timestamp'])
        past_record_ids = history_data['movieId'].values
        print('user past : ',past_record_ids)
        new_data = dataset[dataset['id'].isin(past_record_ids)]
        # print(len(new_data))
        return new_data
    return user_history_documents


def test_custom_env():
    path = '../master_capston/the-movies-dataset/'
    features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
    sampler = LTSDocumentSampler(dataset=features_embedding_movies)

    # this mean the number of items in the recommendation return from the agent
    slate_size = 3

    # i am assuming this number mean the # of possible items to send to the agent for recommend for each slate
    num_candidates = 11

    format_data = data_preprocess.load_data(path)
    # print(format_data.head())
    # print(format_data.shape)

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
        resample_documents=False,offline_mode = True,select_subset_func = func)
    lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)


    observation_0 = lts_gym_env.reset()
    print(observation_0['user'].keys())
    print("current user : ",observation_0['user']['user_id'])
    print("current history of user items :", observation_0['user']['record_ids'])
    print("candidate recommend docs ids : ", observation_0['doc'].keys())
    done = False
    while(not done):
        recommendation_slate_0 = [0, 1, 2]
        observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)
        print("response : ", observation_1['response'])
        print("next history of recommend items :", observation_1['user']['record_ids'])
        print("total remaind candidate items to recommend : ",len(observation_1['doc'].keys()))
        print("docs ids : ", observation_1['doc'].keys())



# test_custom_env()