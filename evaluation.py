# Eyal Kazula 209133693
import math

import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
# Import Pandas
import pandas as pd

def precision_10(test_set, cf, is_user_based = True):
    "*** YOUR CODE HERE ***"
    test_users = test_set.userId.unique()
    top_k_matrix = [cf.predict_movies(user, 10, is_user_based)[0] for user in test_users]
    val = 0 #this value should be changed.
    print("Precision_k: " + str(val))

def ARHA(test_set, cf, is_user_based = True):
    "*** YOUR CODE HERE ***"
    val = 0  # this value should be changed.
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


