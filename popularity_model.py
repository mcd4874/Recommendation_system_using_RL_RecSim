import numpy as np
import pandas as pd
class PopularityRecommender:
    def __init__(self,user_data,slate_size,mode = "0"):
        self.user_data = user_data
        print("total unique movie ",len(np.unique(user_data["movieId"].values)))
        self.slate_size = slate_size
        self.mode = mode
        self.positive_rating_count,self.avg_rating_count = self.generate_popularity_table()
    def generate_popularity_table(self):
        positive_data = self.user_data[self.user_data['rating'] > 3]
        items_positive_data_count = positive_data.groupby('movieId').size().reset_index()
        items_positive_data_count.columns = ['movieId', 'number of positive rating']

        item_data_average_rating = self.user_data[['movieId','rating']].groupby('movieId').mean().reset_index()
        item_data_average_rating.columns = ['movieId','avg_rating']
        return [items_positive_data_count,item_data_average_rating]



    def step(self,reward,observation):
        doc = observation['doc']
        user = observation['user']
        user_id = user['user_id']
        user_past_record_ids = user['record_ids']
        user_past_record = user['past_record']
        doc_ids = list(doc.keys())
        score_list = np.zeros(len(doc_ids))
        # print("doc ids : ",doc_ids)
        if self.mode == "0":
            # print('option 0 ')
            use_dataset = self.positive_rating_count
            # print("total unique movie id : ",len(use_dataset["movieId"].values))
        else:
            # print("option 1")
            use_dataset = self.avg_rating_count
            # print("total unique movie id : ", len(use_dataset["movieId"].values))

        for index in range(len(doc_ids)):
            id = doc_ids[index]
            movie = use_dataset[use_dataset["movieId"] == int(id) ]
            # print("movie info : ",movie)

            movie = movie[use_dataset.columns[-1]].values
            # print("count : ",movie)
            if (len(movie) == 0):
                score_list[index] = -1
            else:
                score_list[index] = movie[0]

        score_list = np.asarray(score_list, dtype=np.float32)
        # print("score lsit ",score_list)
        # print("list of best index : ",score_list.argsort()[::-1])
        return score_list.argsort()[::-1][:self.slate_size]

        # sort_docs = use_dataset[use_dataset["movieId"].isin(doc_ids)].sort_values(by=[use_dataset.columns[-1]],ascending=False)
        # result_doc_ids = sort_docs['movieId'].valuesp[:self.slate_size]
        # result_index = np.where(doc_ids.isin(result_doc_ids))




