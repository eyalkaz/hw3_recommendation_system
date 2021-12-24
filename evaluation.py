# Eyal Kazula 209133693
import math

from sklearn.metrics import mean_squared_error
from math import sqrt
# Import Pandas
import pandas as pd

def precision_10(test_set, cf, is_user_based = True):
    "*** YOUR CODE HERE ***"
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
    test_user_item_matrix = test_set.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if is_user_based:
        prediction_matrix = cf.user_prediction_matrix
    else:
        prediction_matrix = cf.items_prediction_matrix
    for user_id in test_set["userId"].unique():
        for movie_id in get_rated_movies(user_id):
            predicted_rating = prediction_matrix[cf.dict_users_indexes[user_id], cf.dict_movies_indexes[movie_id]]
            actual_rating = test_user_item_matrix.loc[user_id].loc[movie_id]
            numerator += (predicted_rating - actual_rating)**2
    val = math.sqrt(numerator/N)
    print("RMSE: " + str(val))


