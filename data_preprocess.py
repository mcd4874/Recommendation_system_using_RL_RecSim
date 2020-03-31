import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import cosine, correlation
# from surprise import Reader, Dataset, SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
# from surprise.model_selection import cross_validate, KFold ,GridSearchCV , RandomizedSearchCV

from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import  Input, dot, concatenate
from keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from sklearn.model_selection import train_test_split
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

path = '../master_capston/ml-100k/'

# def load_data(sub_amount = 5000):
#     data = pd.read_csv(os.path.join(path,'u.data'), header= None , sep = '\t')
#     user = pd.read_csv(os.path.join(path,'u.user'), header= None , sep = '|')
#     genre = pd.read_csv(os.path.join(path,'u.genre'), header= None , sep = '|' )
#     items = pd.read_csv(os.path.join(path,'u.item') , header = None , sep = "|" , encoding='latin-1')
#     genre.columns = ['Genre' , 'genre_id']
#     data.columns = ['user id' , 'movie id' , 'rating' , 'timestamp']
#     user.columns = ['user id' , 'age' , 'gender' , 'occupation' , 'zip code']
#     items.columns = ['movie id' , 'movie title' , 'release date' , 'video release date' ,
#                   'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
#                   'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
#                   'Film_Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci_Fi' ,
#                   'Thriller' , 'War' , 'Western']
#
#     data = data[:sub_amount]
#     user = user[:sub_amount]
#     genre = genre[:sub_amount]
#     items = items[:sub_amount]
#
#     data = data.merge(user, on='user id')
#     data = data.merge(items, on='movie id')
#     # print(data.head())
#     columns = ['user id' , 'movie id' , 'rating' , 'timestamp','Action' , 'Adventure' , 'Animation' ,
#                   'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
#                   'Film_Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci_Fi' ,
#                   'Thriller' , 'War' , 'Western']
#     return data[columns]

def load_data(path, amount = None):
    rating = pd.read_csv(os.path.join(path,'process_small_rating.csv'))
    if amount:
        return rating[:amount]
    return rating

def get_dataset(data):
    k = 10
    data['rating'] = data['rating'].apply(lambda x: 0 if x < 4 else 1)
    # positive_rate = data[data[data['rating'] == 1].size()>9]
    positive_rate = data[data['rating'] == 1]
    # print(positive_rate)
    user_details = positive_rate.groupby('user id').size().reset_index()
    user_details.columns = ['userid', 'number of positive rating']
    user_details = user_details[user_details['number of positive rating'] > (k-1)]
    # ready_build_dataset = positive_rate[positive_rate['user id'] in user_details['user id']]
    # print(user_details)
    # print(ready_build_dataset)
    return positive_rate

def get_user_positive(data,number_positive_pivot = 30):
    positive_data = data[data['rating'] > 3]
    user_details = positive_data.groupby('userId').size().reset_index()
    user_details.columns = ['userId', 'number of rating']
    user_details = user_details[user_details['number of rating'] > (number_positive_pivot - 1)]
    positive_user_ids = user_details['userId'].values
    return positive_user_ids,data[data['userId'].isin(positive_user_ids)]
def generate_train_test_data(data,ratio=0.2):
    positive_user_ids = np.unique(data['userId'].values)
    first_case = positive_user_ids[0]
    current = data[data['userId'] == first_case]
    final_train_set, final_test_set = train_test_split(current, test_size=ratio)
    for id in positive_user_ids[1:]:
        current = data[data['userId'] == id]
        train_set,test_set = train_test_split(current, test_size=ratio)
        final_train_set = pd.concat([final_train_set, train_set], ignore_index=True)
        final_test_set = pd.concat([final_test_set, test_set], ignore_index=True)
    print("after split")
    print(final_train_set.describe())
    print(final_test_set.describe())
    return [final_train_set,final_test_set]

def create_recent_history(data,embedding_features_data,num_history_items = 10):
    user_ids = np.unique(data['userId'].values)
    # recent_history_dataset = {}
    new_dataset = pd.DataFrame(columns = data.columns)
    recent_history_dataset = pd.DataFrame(columns = data.columns)
    # recent_history = {}
    for id in user_ids:
        # pick one user out of list of possible user to perform the study
        history_embedding_features = np.zeros((num_history_items,embedding_features_data.shape[1]-1))
        history_data = data[data['userId'] == id]
        # print("history data of current user : ",history_data)
        potential_positive_rating = history_data[history_data['rating']>3].sort_values(by=['timestamp'])
        # print("potential positive rating : ",potential_positive_rating)
        positive_history_rating = potential_positive_rating[:num_history_items]
        positive_history_movie_ids = positive_history_rating['movieId'].values
        remainder_history_data = history_data[~history_data['movieId'].isin(positive_history_movie_ids)]
        new_dataset = pd.concat([new_dataset, remainder_history_data], ignore_index=True)
        recent_history_dataset = pd.concat([recent_history_dataset, positive_history_rating], ignore_index=True)



        # for index in range(len(positive_history_movie_ids)):
        #     current_move_id = positive_history_movie_ids[index]
        #     current_movie_embedding_features = embedding_features_data[embedding_features_data['id'] == current_move_id]
        #     history_embedding_features[index] = current_movie_embedding_features.values[0, 1:]
        # recent_history[id] = history_embedding_features

    return [recent_history_dataset,new_dataset]

def generate_csv_file(data,filename):
    data.to_csv(filename)






# format_data = load_data(10000)
# get_dataset(format_data)
# print(format_data.shape)