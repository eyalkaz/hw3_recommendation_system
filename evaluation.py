# Eyal Kazula 209133693
import math

import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
# Import Pandas
import pandas as pd

def precision_10(test_set, cf, is_user_based = True):
    "*** YOUR CODE HERE ***"
    k = 10
    test_user_item_matrix = test_set.pivot(index='userId', columns='movieId', values='rating')
    test_users = test_set.userId.unique()
    top_k_matrix = [cf.predict_movies(user, 10, is_user_based)[0] for user in test_users]
    movies = test_user_item_matrix.columns
    movie_rated_more_then_4_per_user_in_test = [[movies[i] for i, rating in enumerate(test_user_item_matrix.loc[user].values)
                                                if not np.isnan(rating) and rating >= 4] for user in test_users]

    def calc_hits(top, rated_high):  # calculate how many movies appear in both lists
        return set(top) & set(rated_high)  # every movie can appear only one in each list
    hits = [len(calc_hits(top_k_matrix[i], movie_rated_more_then_4_per_user_in_test[i]))/k for i in range(len(test_users))]
    val = sum(hits)/len(test_users)
    print("Precision_k: " + str(val))

def ARHA(test_set, cf, is_user_based = True):
    "*** YOUR CODE HERE ***"
    k = 10
    test_user_item_matrix = test_set.pivot(index='userId', columns='movieId', values='rating')
    test_users = test_set.userId.unique()
    top_k_matrix = [cf.predict_movies(user, 10, is_user_based)[0] for user in test_users]
    movies = test_user_item_matrix.columns
    movie_rated_more_then_4_per_user_in_test = [[movies[i] for i, rating in enumerate(test_user_item_matrix.loc[user].values)
                                                if not np.isnan(rating) and rating >= 4] for user in test_users]

    def calc_hits(top, rated_high):  # calculate how many movies appear in both lists
        return set(top) & set(rated_high)  # every movie can appear only one in each list
    hits = [calc_hits(top_k_matrix[i], movie_rated_more_then_4_per_user_in_test[i]) for i in range(len(test_users))]
    arhr = sum([sum([1/(i+1) for i in range(k) if top_k_matrix[j][i] in hits[j]]) for j in range(len(test_users))])
    val = arhr / len(test_users)
    print("ARHR: " + str(val))

def RSME(test_set, cf, is_user_based = True):
    "*** YOUR CODE HERE ***"
    numerator = 0
    N = test_set.__len__()
    if is_user_based:
        prediction_matrix = cf.user_prediction_matrix
    else:
        prediction_matrix = cf.items_prediction_matrix
    test_users_indexes = [cf.dict_users_indexes[user_id] for user_id in test_set.userId]
    movies_rated_indexes = [cf.dict_movies_indexes[movie_id] for movie_id in test_set.movieId]
    predicted_rating_array = np.array([prediction_matrix[user, movie] for user, movie in zip(test_users_indexes, movies_rated_indexes)])
    add_mean_per_user = cf.mean_user_rating[test_users_indexes]
    actual_rating = test_set.rating.values
    numerator = np.sum((predicted_rating_array - actual_rating)**2)
    val = math.sqrt(numerator/N)
    print("RMSE: " + str(val))


