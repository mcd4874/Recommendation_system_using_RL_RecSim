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
from recsim.choice_model import MultinomialLogitChoiceModel,AbstractChoiceModel
from recsim.simulator import environment
from recsim.simulator.environment import SingleUserEnvironment
from recsim.simulator import recsim_gym
import os
import collections
import random
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
        self.constant_num_candidates = num_candidates

    def check_num_candidate(self):
        if (self._num_candidates > self._document_sampler.size()):
            self._num_candidates = self._document_sampler.size()


    def reset(self):
        """Resets the environment and return the first observation.

        Returns:
          user_obs: An array of floats representing observations of the user's
            current state
          doc_obs: An OrderedDict of document observations keyed by document ids
        """
        self._user_model.reset()
        user_obs = self._user_model.create_observation()
        self._num_candidates = self.constant_num_candidates
        # print("before num candidate ",self._num_candidates)

        # only use a user's history record as recommendable items
        if self.offline_mode and self.select_subset_func is not None:
            user_history_records = self.select_subset_func(user_obs['user_id'])
            if (len(user_history_records)<self._slate_size):
                print("there is problem with current user : ",user_obs['user_id'])
            self._document_sampler.update_dataset(user_history_records,self._num_candidates)
            self.check_num_candidate()
            self._do_resample_documents()
            # print("after num candidate ", self._num_candidates)
            # print("recommendable doc size: ",self._document_sampler.dataset.shape[0])
        elif (not self.offline_mode and self._resample_documents):
            self.check_num_candidate()
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
        self._document_sampler.update_state(documents, responses,self._resample_documents)


        # Obtain next user state observation.
        user_obs = self._user_model.create_observation()

        # Check if reaches a terminal state and return.
        done = self._user_model.is_terminal(remaind_recommendable_size=min(self._candidate_set.size(),self._document_sampler.size()),slate_size=self.slate_size)

        # #update candidate set based on the response
        self.update_candidate_set(documents, responses)

        # Optionally, recreate the candidate set to simulate candidate
        # generators for the next query.
        if self._resample_documents:
            self.check_num_candidate()
            self._do_resample_documents()


        # Create observation of candidate set.
        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation())

        return (user_obs, self._current_documents, responses, done)

    def update_candidate_set(self,documens,responses):
        for doc, response in zip(documens, responses):
            if response.click:
                self._candidate_set.remove_document(doc)

class LTSDocument(document.AbstractDocument):
    def __init__(self, doc_id, embedding_features):
        # print("input doc_id ",doc_id)
        self.embedding_features = embedding_features
        # doc_id is an integer representing the unique ID of this document
        super(LTSDocument, self).__init__(doc_id)

    def create_observation(self):
        return {'doc_id':self._doc_id,'embedding_features':self.embedding_features}


    def observation_space(self):
        # return spaces.Box(shape=(len(self.embedding_features),), dtype=np.float32, low=0.0, high=1.0)
        return spaces.Dict({
            'doc_id': spaces.Discrete(100000),
            'embedding_features':
                spaces.Box(shape=(len(self.embedding_features),), dtype=np.float32, low=0.0, high=1.0)})

    def __str__(self):
        return "Document {} ".format(self._doc_id)



class LTSDocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, dataset ,num_candidate,doc_ctor=LTSDocument, **kwargs):
        super(LTSDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self.dataset = dataset
        self.count = 0
        # self.num_candidate = num_candidate
        self.max_num_recommendable_ids =num_candidate
        self.recommendable_doc_ids = self.dataset[self.dataset.columns[0]].values
        self.generate_list_recommendable_items()

    def generate_list_recommendable_items(self):
        doc_ids = self.dataset[self.dataset.columns[0]].values
        self.max_num_recommendable_ids = min(self.max_num_recommendable_ids,len(doc_ids))
        recommendable_doc_ids = np.random.choice(doc_ids, self.max_num_recommendable_ids, replace=False)
        self.recommendable_doc_ids = recommendable_doc_ids
        self.count = 0

    def sample_document(self):
        columns_id = self.dataset.columns[0]
        current_item_index = self.recommendable_doc_ids[self.count]
        current_item = self.dataset[self.dataset[columns_id] == current_item_index].values.flatten()
        if len(current_item) == 0:
            print("can not find this item in the dataset ")
            print(" the current ids in the dataset : ",self.dataset[columns_id].values)
            print("the current ids in recommendable list : ",self.recommendable_doc_ids)
        doc_features = {}
        doc_features['doc_id'] = int(current_item[0])
        doc_features['embedding_features'] = current_item[1:]
        self.count = (self.count+1)

        # generate a new list of recommendable items after every resample step
        if self.count == self.max_num_recommendable_ids:
            self.generate_list_recommendable_items()


        return self._doc_ctor(**doc_features)

    def update_state(self, documents, responses,num_candidate = None,resampled = True):
        """Update document state (if needed) given user's (or users') responses.
            remove those document (item spaces )

        """
        #remove the documents that user selected in recommendable dataset
        list_id = list()
        id_col = self.dataset.columns[0]
        for index in range(len(documents)):
            doc = documents[index]
            response = responses[index]
            if response.click:
                doc_obs = doc.create_observation()
                list_id.append(doc_obs['doc_id'])


        self.dataset = self.dataset[~self.dataset[id_col].isin(list_id)]
        # print("new dataset size in doc sampler update : ",len(self.dataset))
        # print(self.dataset[id_col].values)

        # #remove the documets in the recommendable list
        # self.recommendable_doc_ids

        #start again at the beginning of the current recommendable list
        # self.count = 0
        if num_candidate is not None:
            self.num_candidate = num_candidate

        if resampled:
            self.generate_list_recommendable_items()

        #TODO: deal with case when we have to remove items in recommendable list IDs, not resample



    def update_dataset(self,dataset,num_candidate= None):
        self.dataset = dataset
        # print("total number of items belong to this user : ",len(self.dataset))
        self.count = 0
        if num_candidate is not None:
            self.max_num_recommendable_ids = num_candidate
        self.generate_list_recommendable_items()

    def size(self):
        return len(self.dataset)

class LTSUserState(user.AbstractUserState):
    def __init__(self, memory_discount,time_budget,user_info,offline_mode = True, offline_data = None ,corpus_features_dim = 30,history_record_size = 10):
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

        self.offline_mode = offline_mode
        self.user_offline_record = offline_data


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
        doc_id = doc_obs['doc_id']
        embedding_features = doc_obs['embedding_features']
        if self.offline_mode:
            match_record_rating = self.user_offline_record[(self.user_offline_record['userId'] == self.user_id) & (self.user_offline_record['movieId'] == doc_id)]['rating']
            if (len(match_record_rating) == 1):
                # print("score properly")
                # print(match_record_rating.values)
                return match_record_rating.values

        return 0
        #jsut sum all the features and avergae for now
        # likely = np.sum(embedding_features)
        # return 1 - likely

    def update_time_buget(self):
        self.time_budget -=1

    def is_offline(self):
        return self.offline_mode
    def get_user_offline_record(self):
        return self.user_offline_record

    def update_user_history_record(self,new_doc_id, new_doc_feature):
        n = len(self.user_recent_past_record_ids)
        for index in range(1,n):
            self.user_recent_past_record_ids[index-1] = self.user_recent_past_record_ids[index]
            self.user_recent_past_record[index-1] = self.user_recent_past_record[index]
        self.user_recent_past_record_ids[n-1] = new_doc_id
        self.user_recent_past_record[n - 1] = new_doc_feature


class LTSStaticUserSampler(user.AbstractUserSampler):
    def __init__(self, user_recent_history_data,corpus_data,offline_data = None,offline_mode = True,memory_discount=0.9,
               time_budget=4,history_size = 10,doc_feature_size = 30,
               user_ctor=LTSUserState,seed = 0,random = True,time_budget_range = None,
               **kwargs):
        super(LTSStaticUserSampler, self).__init__(user_ctor, **kwargs)

        self.corpus_data = corpus_data
        self.user_recent_history_data = user_recent_history_data
        self.posible_user_ids = np.unique(self.user_recent_history_data['userId'].values)
        self.seed = seed
        self.doc_feature_size = doc_feature_size
        self.history_size = history_size

        self.offline_mode = offline_mode
        self.offline_data = offline_data
        self.time_budget_range = time_budget_range
        self._state_parameters = {'memory_discount': memory_discount,
                                  'time_budget': time_budget,
                                  'offline_mode':self.offline_mode,
                                  'offline_data':self.offline_data
                                  }
        self.random = random
        self.current_user_index = 0

    def sample_user(self):
        if (self.random):
            pick_user_id = np.random.choice(self.posible_user_ids, 1)[0]
        else:
            #go through all of a user in the database ( shuffle everytime we go through the list
            if (self.current_user_index == 0):
                np.random.shuffle(self.posible_user_ids)
            pick_user_id = self.posible_user_ids[self.current_user_index]
            self.current_user_index = (self.current_user_index+1)%len(self.posible_user_ids)

        # pick one user out of list of possible user to perform the study

        history_data = self.user_recent_history_data[self.user_recent_history_data['userId'] == pick_user_id].sort_values(by=['timestamp'])

        # create a matrix for user history doc features
        past_record = np.zeros((self.history_size,self.doc_feature_size))
        past_record_ids = history_data['movieId'].values
        for index in range(self.history_size):
            current_move_id = past_record_ids[index]
            current_movie_embedding_features = self.corpus_data[self.corpus_data['id'] == current_move_id]
            past_record[index] = current_movie_embedding_features.values[0,1:]


        user_info = {
            'userId': pick_user_id,
            'record_ids': past_record_ids,
            'past_record': past_record
        }
        if self.time_budget_range is not None:
            random_time_budget = random.randint(self.time_budget_range[0],self.time_budget_range[1]+1)
            self._state_parameters['time_budget'] = random_time_budget

        self._state_parameters['user_info'] = user_info
        return self._user_ctor(**self._state_parameters)


class LTSResponse(user.AbstractResponse):
    # The maximum degree of engagement.
    MAX_ENGAGEMENT_MAGNITUDE = 100.0

    def __init__(self, click=False, engagement=0.0,rating = -1):
        self.click = click
        self.engagement = engagement
        self.rating = rating

    def create_observation(self):
        return {'click': int(self.click), 'engagement': np.array(self.engagement),'rating':int(self.rating)}

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
                    dtype=np.float32),
            'rating':
            spaces.Discrete(6),
        })
    def update_response(self,click,rating,engagement):
        self.click = click
        self.engagement = engagement
        self.rating = rating

class AbstractRatingModel(object):
    """Abstract class to represent the user rating model.

    Each user has a rating model.
    """
    _scores = None

    def score_documents(self, user_state, doc_obs):
        """Computes unnormalized scores of documents in the slate given user state.

        Args:
          user_state: An instance of AbstractUserState.
          doc_obs: A numpy array that represents the observation of all documents in
            the slate.
        Attributes:
          scores: A numpy array that stores the scores of all documents.
        """

    @property
    def scores(self):
        return self._scores

    def choose_item(self,rating_pivot):
        """Returns selected index of documents in the slate such that the rating >= rating_pivot.
        Returns:
          selected_indexs: list og integer indicating which items were chosen, or None if
            there were no items.
        """
class HistoryChoiceModel(AbstractRatingModel):
    """A historical choice model.
    Args:
     choice_features: a dict that stores the features used in choice model:
       `no_click_mass`: a float indicating the mass given to a no click option.
    """
    def __init__(self, choice_features):
        self._no_click_mass = choice_features.get('no_click_mass', -float('Inf'))

    def score_documents(self, user_state, doc_obs):
        scores = np.array([])
        if user_state.is_offline():
            user_id = user_state.create_observation()['user_id']
            for doc in doc_obs:
                scores = np.append(scores, user_state.score_document(doc))
            self._scores = scores
    def choose_item(self,rating_pivot):
        select_index_list = np.argwhere(self._scores >= rating_pivot)
        return select_index_list





class UserModel(user.AbstractUserModel):
    def __init__(self, sampler, offline_mode = True ,rating_pivot = 4,slate_size = 10,response_ctor = LTSResponse):
        super(UserModel, self).__init__(response_ctor,sampler,slate_size)
        self.offline_mode = offline_mode
        self.rating_pivot = rating_pivot
        if (self.offline_mode):
            # print("use history model")
            self.response_model = HistoryChoiceModel({})
        else:
            self.response_model = MultinomialLogitChoiceModel({})


    def simulate_response(self, slate_documents):
        # List of empty responses
        responses = [self._response_model_ctor() for _ in slate_documents]
        # Get click from of choice model.
        self.response_model.score_documents(
            self._user_state, [doc.create_observation() for doc in slate_documents])

        scores = self.response_model.scores
        # print("possible scores : ",scores)
        for index in range(len(scores)):
            self.generate_response(slate_documents[index],responses[index],scores[index])
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
            if response.click:
                self._user_state.satisfaction = 0
                doc_obs = doc.create_observation()
                doc_id = doc_obs['doc_id']
                doc_features = doc_obs['embedding_features']
                self._user_state.update_user_history_record(doc_id,doc_features)

        self._user_state.update_time_buget()



    def is_terminal(self,remaind_recommendable_size = -1,slate_size = -1):
        """Returns a boolean indicating if the session is over."""
        if (remaind_recommendable_size < slate_size or self._user_state.time_budget <= 0):

            return True
        return False

    def generate_response(self, doc, response,rating_score):
        clicked = False
        if (rating_score>= self.rating_pivot):
            clicked = True
        # print("response click : ",clicked)
        engagement = 0
        response.update_response(clicked,rating_score,engagement)




def clicked_engagement_reward(responses):
    """
    this function will calculate the reward
    :param responses:
    :return:
    """
    reward = 0.0
    total = len(responses)
    for response in responses:
        reward = reward+((response.rating-3)/2.0)

    #normalize rating between -1 and 1
    # reward = reward

    return reward/total

def select_dataset(dataset,user_data):
    def user_history_documents(user_id):
        history_data = user_data[user_data['userId'] == user_id]
        past_record_ids = history_data['movieId'].values
        # print('user past : ',past_record_ids)
        new_data = dataset[dataset['id'].isin(past_record_ids)]
        # print(len(new_data))
        return new_data
    return user_history_documents


def test_custom_env():
    path = '../master_capston/the-movies-dataset/'
    features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))

    # this mean the number of items in the recommendation return from the agent
    slate_size = 3

    # i am assuming this number mean the # of possible items to send to the agent for recommend for each slate
    num_candidates = 11

    format_data = data_preprocess.load_data(path)
    features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
    positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data)
    # generate train and test set
    train_set, test_set = data_preprocess.generate_train_test_data(positive_history_data)
    users_history_data, train_set = data_preprocess.create_recent_history(train_set,
                                                                          embedding_features_data=features_embedding_movies)

    offline_mode = True
    rating_pivot = 4

    sampler = LTSDocumentSampler(dataset=features_embedding_movies,num_candidate=num_candidates)
    user_sampler = LTSStaticUserSampler(users_history_data ,features_embedding_movies,offline_data=test_set,offline_mode=offline_mode)
    # need to handle where we update dataset with num candidate< available
    func = select_dataset(features_embedding_movies, test_set)
    LTSUserModel = UserModel(user_sampler, offline_mode=offline_mode,rating_pivot=rating_pivot,slate_size=slate_size, response_ctor=LTSResponse)
    ltsenv = CustomSingleUserEnviroment(
        LTSUserModel,
        sampler,
        num_candidates,
        slate_size,
        resample_documents=False, offline_mode=True, select_subset_func=func)
    lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)


    observation_0 = lts_gym_env.reset()
    print("current user : ",observation_0['user']['user_id'])
    print("current history of user items :", observation_0['user']['record_ids'])
    print("candidate recommend docs ids : ", observation_0['doc'].keys())
    done = False
    while(not done):
    # for i in range(4):
        recommendation_slate_0 = [0, 1, 2]
        observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)
        print("response : ", observation_1['response'])
        print("reward : ",reward)
        print("next history of recommend items :", observation_1['user']['record_ids'])
        print("total remaind candidate items to recommend : ",len(observation_1['doc'].keys()))
        print("docs ids : ", observation_1['doc'].keys())




# test_custom_env()