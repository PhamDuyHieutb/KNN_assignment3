"""
Hieu Pham winter 2018

"""

import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import numpy as np

# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1


def RMSE(n_test_users, n_remaining_movies, true_ratings_matrix, predicted_ratings_matrix):
    print("Computing RMSE")
    squared_error = 0.0
    n_tests = 0
    for user_id in range(n_test_users):
        nonzero_ratings = []
        for movie_id in range(n_remaining_movies):
            if true_ratings[user_id, movie_id] > 0.0:
                nonzero_ratings.append(movie_id)

        squared_error += np.sum(
            (true_ratings_matrix[user_id, nonzero_ratings] - predicted_ratings_matrix[user_id, nonzero_ratings]) ** 2)
        n_tests += len(nonzero_ratings)

    RMSE = np.sqrt(squared_error / n_tests)

    return RMSE

def read_data(path):
    user_ids = []
    movie_ids = []
    ratings = []
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    data = pd.read_csv(path, sep='\t', names=r_cols).values

    for user_id, movie_id, rating, timestamp in data:
        user_ids.append(int(user_id))
        movie_ids.append(int(movie_id))
        ratings.append(float(rating))

    n_movies = max(movie_ids) + 1
    n_users = max(user_ids) + 1

    ratings_matrix = np.zeros((n_users, n_movies))

    for user_id, movie_id, rating in zip(user_ids, movie_ids, ratings):
        ratings_matrix[user_id, movie_id] = rating

    return ratings_matrix



def MAE(n_test_users, n_remaining_movies, true_ratings_matrix, predicted_ratings_matrix):
    print("Computing MAE")
    absolute_error = 0.0
    n_tests = 0
    for user_id in range(n_test_users):
        nonzero_ratings = []
        for movie_id in range(n_remaining_movies):
            if true_ratings[user_id, movie_id] > 0.0:
                nonzero_ratings.append(movie_id)

        absolute_error += np.sum(
            abs(true_ratings_matrix[user_id, nonzero_ratings] - predicted_ratings_matrix[user_id, nonzero_ratings]))
        n_tests += len(nonzero_ratings)

    MAE = absolute_error / n_tests

    return MAE

def NMAE(MAE, predicted_ratings_matrix):
    arr = []
    for k in predicted_ratings_matrix:
        for i in k:
            arr.append(i)

    NMAE_score = MAE / (max(arr) - min(arr))
    return NMAE_score



train_rating_file = "ml-100k/u.data"
test_rating_file = "ml-100k/u.test"
k = 100

# read data
print ("Reading data ratings")
ratings_matrix = read_data(train_rating_file)

num_users = ratings_matrix.shape[0]
num_movies = ratings_matrix.shape[1]
print("num users = " + str(num_users))
print("num movies " + str(num_movies))
num_training_users = int(num_users * 0.8)

print ("Split train test")
train_ids = random.sample(range(num_users),
                          num_training_users)
test_ids = set(range(num_users)) - set(train_ids)
test_ids = list(test_ids)
num_test_users = len(test_ids)

training_matrix = ratings_matrix[train_ids, :]
testing_matrix = ratings_matrix[test_ids, :]
true_ratings = testing_matrix.copy()

# using Impute for imputing missing values
print ("Impute values for missing value")
imputer = SimpleImputer(missing_values=0)

training_imputed_matrix = imputer.fit_transform(training_matrix)
testing_imputed_matrix = imputer.transform(testing_matrix)

# convert to sparse matrix => save memory and computation
selected_columns = []
for movie_id in range(num_movies):
    if not np.isnan(imputer.statistics_[movie_id]):
        selected_columns.append(movie_id)

training_matrix = training_matrix[:, selected_columns]
testing_matrix = testing_matrix[:, selected_columns]
true_ratings = true_ratings[:, selected_columns]

n_remaining_movies = training_matrix.shape[1]

#  fit model KNN
print (" fitting model KNN")
knn = NearestNeighbors()
knn.fit(training_imputed_matrix)

# returns num_test_users x k matrix for next step
neighbor_indices = knn.kneighbors(testing_imputed_matrix, n_neighbors= k, return_distance=False)

# compute average ratings for each user
print ("Computing average ratings")
predicted_ratings = np.zeros((num_test_users,
                              n_remaining_movies))

for user_id in range(num_test_users):
    neighbors = neighbor_indices[user_id, :]
    predicted_ratings[user_id, :] = np.average(training_imputed_matrix[neighbors, :], axis=0)


RMSE_score = RMSE(num_test_users, n_remaining_movies, true_ratings, predicted_ratings)
print ("Root Mean-Squared Error:", RMSE_score)

MAE_score  = MAE(num_test_users, n_remaining_movies, true_ratings, predicted_ratings)
print("Mean Absolute Error: ", MAE_score)

NMAE_score = NMAE(MAE_score, predicted_ratings)
print("Normalized Mean Absolute Error: ", NMAE_score)
