import numpy as np
import pandas as pd
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym
import os

import data_preprocess

class LTSDocument(document.AbstractDocument):
    def __init__(self, doc_id, embedding_features):
        # print("input doc_id ",doc_id)
        self.embedding_features = embedding_features
        # doc_id is an integer representing the unique ID of this document
        super(LTSDocument, self).__init__(doc_id)

    def create_observation(self):
        return self.embedding_features
        # return np.array([self.kaleness])


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


        # for row in self.dataset.rows:
        #     doc_features = {}
        #     doc_features['movie_id'] = row[columns[0]]
        #     doc_features['embedding'] = row[columns[1:]]
        return self._doc_ctor(**doc_features)

class LTSUserState(user.AbstractUserState):
    def __init__(self, memory_discount,time_budget,user_info):
        ## Transition model parameters
        ##############################
        # self.memory_discount = memory_discount
        # self.sensitivity = sensitivity
        # self.innovation_stddev = innovation_stddev


        ## State variables
        self.time_budget = time_budget
        self.satisfaction = 0
        self.corpus_features_dim = 30
        self.slate_size = 10

        self.user_id = user_info['userId']
        self.user_recent_past_record_ids = user_info['record_ids']
        self.user_recent_past_record = user_info['past_record']

        # update the user recent history with new recommendation from left to right
        self.state_update_index = 0


    def create_observation(self):
        """User's state is not observable."""
        obs = np.array(self.user_recent_past_record)
        return obs

    def observation_space(self):
        return spaces.Box(shape=(self.slate_size,self.corpus_features_dim), dtype=np.float32, low=-10.0, high=10.0)

    # scoring function for use in the choice model -- the user is more likely to
    # click on more chocolatey content.
    def score_document(self, doc_obs):
        # print('doc obs :',doc_obs)

        #jsut sum all the features and avergae for now
        likely = np.sum(doc_obs)
        return 1 - likely

    def update_time_buget(self):
        self.time_budget -=1

    def get_time_buget(self):
        return self.time_budget

    def update_user_history_record(self,new_doc_id, new_doc_feature):
        self.user_recent_past_record_ids[self.state_update_index]= new_doc_id
        self.user_recent_past_record[self.state_update_index] = new_doc_feature
        self.state_update_index = (self.state_update_index+1) % len(self.user_recent_past_record)
    def get_user_history_record_ids(self):
        return self.user_recent_past_record_ids


class LTSStaticUserSampler(user.AbstractUserSampler):
  # _state_parameters = None

    def __init__(self, posible_user_ids,user_data,corpus_data,memory_discount=0.9,
               time_budget=10,
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






    def sample_user(self):
        state_size = 10
        doc_feature_size = 30

        # pick one user out of list of possible user to perform the study
        pick_user_id = np.random.choice(self.posible_user_ids, 1)[0]
        history_data = self.user_data[self.user_data['userId'] == pick_user_id].sort_values(by=['timestamp'])
        while(history_data.shape[0] < state_size ):
            print('need to resample')
            pick_user_id = np.random.choice(self.posible_user_ids, 1)[0]
            print('user id resample',pick_user_id)
            history_data = self.user_data[self.user_data['userId'] == pick_user_id].sort_values(by=['timestamp'])


        # create a matrix for user history doc features
        past_record = np.zeros((state_size,doc_feature_size))
        past_record_ids = history_data['movieId'].values[:state_size]
        for index in range(state_size):
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
        print("the select corpus index is :",selected_index)
        print("length of slate doc :",len(slate_documents))
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
  reward = 0.0
  for response in responses:
    if response.clicked:
      reward += response.engagement
  return reward



def test_doc_model():
    path = '../master_capston/the-movies-dataset/'
    features_embedding_movies = pd.read_csv(os.path.join(path,'movie_embedding_features.csv'))
    sampler = LTSDocumentSampler(dataset=features_embedding_movies)
    for i in range(5):
        doc = sampler.sample_document()
        # print(doc.doc_id())
        print(doc.create_observation().shape)
    # d = sampler.sample_document()
    # print("Documents have observation space:", d.observation_space(), "\n"
    #                                                                   "An example realization is: ",
    # d.create_observation())
    # a = d.create_observation()
    # print(a[])


def test_user_model():
    path = '../master_capston/the-movies-dataset/'
    format_data = data_preprocess.load_data(path)
    # print(format_data.head())
    # print(format_data.shape)

    features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
    positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data)
    user_sampler = LTSStaticUserSampler(positive_user_ids,positive_history_data,features_embedding_movies)
    current_user = user_sampler.sample_user()
    current_user.create_observation()
    # print(positive_user_ids)
    # print(positive_history_data.head())

def test_env():
    path = '../master_capston/the-movies-dataset/'
    features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
    sampler = LTSDocumentSampler(dataset=features_embedding_movies)
    slate_size = 3
    num_candidates = 10

    format_data = data_preprocess.load_data(path)
    # print(format_data.head())
    # print(format_data.shape)

    features_embedding_movies = pd.read_csv(os.path.join(path, 'movie_embedding_features.csv'))
    positive_user_ids, positive_history_data = data_preprocess.get_user_positive(format_data)
    user_sampler = LTSStaticUserSampler(positive_user_ids, positive_history_data, features_embedding_movies)



    LTSUserModel = UserModel(user_sampler,slate_size,LTSResponse)

    ltsenv = environment.Environment(
        LTSUserModel,
        sampler,
        num_candidates,
        slate_size,
        resample_documents=True)
    lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)

    observation_0 = lts_gym_env.reset()
    print(observation_0['user'][:5])
    # print('Observation 0')
    # print('Available documents')
    # doc_strings = ['doc_id ' + key + " kaleness " + str(value) for key, value
    #                in observation_0['doc'].items()]
    # print('\n'.join(doc_strings))

    recommendation_slate_0 = [0, 1, 2]
    observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)

    print(observation_1['user'][:5])
    # print('Noisy user state observation')
    # print(observation_0['user'])

# test_doc_model()

# test_user_model()
test_env()