# Eyal Kazula 209133693
import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []
        self.user_item_matrix = []
        self.movies_rating_matrix = []
        self.mean_user_rating = []
        self.dict_users_indexes = {}
        self.dict_indexes_users = {}
        self.dict_indexes_movies = {}
        self.dict_movies_indexes = {}
        self.movie_titles = {}
        self.user_prediction_matrix = []
        self.items_prediction_matrix = []
    def create_fake_user(self,rating):
        "*** YOUR CODE HERE ***"
        self.calc_method_matrix(rating)
        self.user_item_matrix.loc[283238] = 0
        self.mean_user_rating = np.append(self.mean_user_rating, [0])
        self.mean_user_rating = self.mean_user_rating.reshape(self.mean_user_rating.shape[0], 1)
        self.calc_matrix_similarity()
        return self.predict_movies(283238, 5)

    def create_movies_title_dict(self, df):
        self.movie_titles = dict(zip(df.movieId, df.title))

    def calc_method_matrix(self, ratings, is_user=True):
        if is_user:
            self.user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
            self.mean_user_rating = self.user_item_matrix.mean(axis=1).to_numpy().reshape(-1, 1)
            self.user_item_matrix = self.user_item_matrix.sub(self.user_item_matrix.mean(axis=1), axis=0).fillna(0)
            self.movies_rating_matrix = self.user_item_matrix.T
        else:
            self.movies_rating_matrix = ratings.pivot(index='movieId', columns='userId', values='rating')
            self.mean_user_rating = self.movies_rating_matrix.T.mean(axis=1).to_numpy().reshape(-1, 1)
            # we take the mean of the columns now because we need mean of users
            self.movies_rating_matrix = self.movies_rating_matrix.sub(self.movies_rating_matrix.mean(axis=0),
                                                                      axis=1).fillna(0)
            self.user_item_matrix = self.movies_rating_matrix.T

        self.dict_users_indexes = {k: v for v, k in enumerate(self.user_item_matrix.index)}  # flipped dictionary with user : index
        self.dict_indexes_users = dict(enumerate(self.user_item_matrix.index))
        self.dict_movies_indexes = {k: v for v, k in enumerate(self.movies_rating_matrix.index)}
        self.dict_indexes_movies = dict(enumerate(self.movies_rating_matrix.index))  # index : movie

    def calc_matrix_similarity(self, is_user=True):
        if is_user:
            sim_matrix = 1 - pairwise_distances(self.user_item_matrix, metric='cosine')  # cosine similarity = 1 - cosiine distance
            self.user_based_matrix = sim_matrix
        else:
            sim_matrix = 1 - pairwise_distances(self.movies_rating_matrix, metric='cosine')  # cosine similarity = 1 - cosiine distance
            self.item_based_metrix = sim_matrix

    def create_prediction_matrix(self, is_user=True):
        if is_user:
            prediction_matrix = self.user_based_matrix.dot(self.user_item_matrix)
            denominator = np.array([np.abs(self.user_based_matrix).sum(axis=1)]).T
            prediction_matrix = (prediction_matrix / denominator)
            prediction_matrix = self.mean_user_rating + np.nan_to_num(prediction_matrix)
            self.user_prediction_matrix  = prediction_matrix
        else:
            prediction_matrix = self.item_based_metrix.dot(self.movies_rating_matrix)
            denominator = np.array([np.abs(self.item_based_metrix).sum(axis=1)]).T
            prediction_matrix = (prediction_matrix / denominator).T  # transpose so it predict for user
            prediction_matrix = self.mean_user_rating + np.nan_to_num(prediction_matrix)
            self.items_prediction_matrix = prediction_matrix

    def create_user_based_matrix(self, data):
        ratings = data[0]

        # for adding fake user
        #ratings = self.create_fake_user(ratings)

        "*** YOUR CODE HERE ***"
        self.create_movies_title_dict(data[1])
        self.calc_method_matrix(ratings, True)
        self.calc_matrix_similarity(True)
        self.create_prediction_matrix(True)

    def create_item_based_matrix(self, data):
        "*** YOUR CODE HERE ***"
        ratings = data[0]
        self.create_movies_title_dict(data[1])
        self.calc_method_matrix(ratings, False)
        self.calc_matrix_similarity(False)
        self.create_prediction_matrix(False)

    def predict_movies(self, user_id, k, is_user_based=True):
        "*** YOUR CODE HERE ***"
        if is_user_based:
            prediction_matrix = self.user_prediction_matrix
        else:
            prediction_matrix = self.items_prediction_matrix

        user_id = int(user_id)
        candidates = prediction_matrix[self.dict_users_indexes[user_id]]
        already_rated = np.where(self.user_item_matrix.loc[[user_id]].values != 0)[1]
        candidates[already_rated] = min(candidates) - 1  # never choose them
        top_k = np.argpartition(candidates, -k)[-k:].tolist()

        def my_func(x):
            return candidates[x]
        top_k.sort(reverse=True, key=my_func)
        top_k_movies_id = [self.dict_indexes_movies[i] for i in top_k]
        top_k_movies_titles = [self.movie_titles[movie_id] for movie_id in top_k_movies_id]
        return top_k_movies_id, top_k_movies_titles

