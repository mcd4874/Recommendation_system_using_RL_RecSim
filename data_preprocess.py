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

def get_user_positive(data):
    k = 10
    data['rating'] = data['rating'].apply(lambda x: 0 if x < 4 else 1)
    # positive_rate = data[data[data['rating'] == 1].size()>9]
    positive_rate = data[data['rating'] == 1]
    # print(positive_rate)
    user_details = positive_rate.groupby('userId').size().reset_index()
    user_details.columns = ['userId', 'number of positive rating']
    user_details = user_details[user_details['number of positive rating'] > (k - 1)]
    # ready_build_dataset = positive_rate[positive_rate['user id'] in user_details['user id']]
    # print(user_details)
    # print(ready_build_dataset)
    user_details = user_details.reset_index()
    return user_details['userId'].values, positive_rate[['userId','movieId','rating','timestamp']]

# format_data = load_data(10000)
# get_dataset(format_data)
# print(format_data.shape)