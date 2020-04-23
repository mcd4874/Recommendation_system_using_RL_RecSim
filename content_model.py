import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class contentModel:
    def __init__(self,slate_size,embedding_size):
        self.slate_size= slate_size
        self.embedding_size = embedding_size
        print("init")
    def generate_recommendation(self,user_history,item_spaces):
        """
        :param user_history: (hxm) where m is embedding size
        :param item_spaces: (nxm)
        :return:
        """
        #size result (hxn)
        calculate_table = cosine_similarity(user_history,item_spaces)
        #reduce to (1xn) by sum all row
        result = np.sum(calculate_table,axis=0)
        result = result.flatten()
        recommended_indexs = result.argsort()[::-1][:self.slate_size]
        return recommended_indexs
    def generate_doc_representation(self, doc):
        corpus = []
        origin_corpus = list(doc.values())
        for doc_obs in origin_corpus:
            doc_features = doc_obs['embedding_features']
            corpus.append(doc_features)
        corpus = np.array(corpus).reshape(len(origin_corpus), self.embedding_size)
        return corpus
    def step(self,reward,observation):
        doc = observation['doc']
        user = observation['user']
        user_past_record = user['past_record']
        items_space = self.generate_doc_representation(doc)
        recommend_index = self.generate_recommendation(user_past_record,items_space)
        return recommend_index